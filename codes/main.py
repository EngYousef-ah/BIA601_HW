import argparse
import csv
import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAMBDA = 0.1        

# اسم الحجم : (عدد المهام n , عدد الموظفين m , ميزانية التقييمات)
SIZES = {
    "Small":  (10, 3, 4000),
    "Medium": (50, 8, 15000),
    "Large":  (200, 20, 40000),
}


def make_problem(n, m, seed=42):
    """يولّد مصفوفة الزمن T[i][j] = الساعات التي يحتاجها الموظف i لإنجاز المهمة j."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1, 10, n)           # الزمن الأساسي للمهمة (ساعات)
    speed = rng.uniform(0.7, 1.3, m)       # سرعة كل موظف
    skill = rng.uniform(0.6, 1.6, (m, n))  # ملاءمة الموظف للمهمة (أقل = أنسب)
    return skill * base[None, :] / speed[:, None]


def fitness(X, T):
    m, n = T.shape
    t = T[X, np.arange(n)]                          # زمن كل مهمة عند موظفها
    loads = np.stack([(t * (X == i)).sum(1) for i in range(m)], axis=1)
    return loads.max(1) + LAMBDA * loads.mean(1)



def ga(T, budget, rng, pop=50, pc=0.9, k=3, elite=2):
    m, n = T.shape
    pm = 1.0 / n                                     
    X = rng.integers(0, m, (pop, n))
    f = fitness(X, T)
    evals = pop
    hist = [(evals, f.min())]
    for _ in range(budget // pop - 1):
        w1 = rng.integers(0, pop, (pop, k)); w1 = w1[np.arange(pop), f[w1].argmin(1)]
        w2 = rng.integers(0, pop, (pop, k)); w2 = w2[np.arange(pop), f[w2].argmin(1)]
        A, B = X[w1], X[w2]
        mask = rng.random((pop, n)) < 0.5
        child = np.where(mask, A, B)
        no_cross = rng.random(pop) >= pc
        child[no_cross] = A[no_cross]
        mut = rng.random((pop, n)) < pm
        child[mut] = rng.integers(0, m, mut.sum())
        fc = fitness(child, T)  
        evals += pop
        old_best = np.argsort(f)[:elite]
        worst = np.argsort(fc)[-elite:]
        child[worst], fc[worst] = X[old_best], f[old_best]
        X, f = child, fc
        hist.append((evals, f.min()))
    b = f.argmin()
    return X[b], f[b], hist


def pso(T, budget, rng, swarm=30, w=0.7, c1=1.5, c2=1.5):
    m, n = T.shape
    vmax = m / 2.0

    def decode(pos):                                
        return np.minimum(np.clip(pos, 0, m - 1e-9).astype(int), m - 1)

    pos = rng.random((swarm, n)) * m
    vel = rng.uniform(-1, 1, (swarm, n))
    f = fitness(decode(pos), T)
    pbest, pf = pos.copy(), f.copy()
    g = pf.argmin(); gbest, gf = pbest[g].copy(), pf[g]
    evals = swarm
    hist = [(evals, gf)]
    for _ in range(budget // swarm - 1):
        r1, r2 = rng.random((swarm, n)), rng.random((swarm, n))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, 0, m - 1e-9)
        f = fitness(decode(pos), T)
        evals += swarm
        imp = f < pf
        pbest[imp], pf[imp] = pos[imp], f[imp]
        g = pf.argmin()
        if pf[g] < gf:
            gbest, gf = pbest[g].copy(), pf[g]
        hist.append((evals, gf))
    return decode(gbest[None, :])[0], gf, hist


def aco(T, budget, rng, ants=20, alpha=1.0, beta=2.0, rho=0.1,
        tau_min=0.1, tau_max=10.0):
    m, n = T.shape
    tau = np.ones((n, m))                       
    q = fitness(greedy(T)[None, :], T)[0]         
    gx, gf = None, np.inf
    evals = 0
    hist = []
    ar = np.arange(ants)
    for _ in range(budget // ants):
        loads = np.zeros((ants, m))
        X = np.zeros((ants, n), dtype=int)
        for j in rng.permutation(n):                 
            eta = 1.0 / (loads + T[:, j][None, :])   
            p = (tau[j][None, :] ** alpha) * (eta ** beta)
            p /= p.sum(1, keepdims=True)
            c = p.cumsum(1)
            ch = np.minimum((rng.random((ants, 1)) > c).sum(1), m - 1)
            X[:, j] = ch
            loads[ar, ch] += T[ch, j]
        f = fitness(X, T)
        evals += ants
        ib = f.argmin()
        if f[ib] < gf:
            gx, gf = X[ib].copy(), f[ib]

        tau *= (1 - rho)
        tau[np.arange(n), X[ib]] += q / f[ib]
        tau[np.arange(n), gx] += q / gf
        tau = np.clip(tau, tau_min, tau_max)
        hist.append((evals, gf))
    return gx, gf, hist


def greedy(T):
    # الأطول أولاً: كل مهمة تٌسند للموظف الذي يعطي أقل حمل بعد الإسناد
    m, n = T.shape
    loads = np.zeros(m)
    x = np.zeros(n, dtype=int)
    for j in np.argsort(-T.min(0)):
        i = (loads + T[:, j]).argmin()
        x[j] = i
        loads[i] += T[i, j]
    return x


def random_search(T, budget, rng):
    m, n = T.shape
    best, hist, evals = np.inf, [], 0
    chunk = 100
    while evals < budget:
        X = rng.integers(0, m, (chunk, n))
        best = min(best, fitness(X, T).min())
        evals += chunk
        hist.append((evals, best))
    return None, best, hist


def brute_force(T):
    # الحل الأمثل بالبحث الشامل

    m, n = T.shape
    X = np.array(list(itertools.product(range(m), repeat=n)), dtype=np.int8)
    f = np.concatenate([fitness(X[i:i + 20000].astype(int), T)
                        for i in range(0, len(X), 20000)])
    return f.min()



ALGOS = {"GA": ga, "PSO": pso, "ACO": aco, "Random": random_search}


def to_curve(hist, budget, points=200):
    # يحول السجل إلى منحنى على محور تقييمات موحّد للمقارنة

    ev = np.array([h[0] for h in hist]); fv = np.array([h[1] for h in hist])
    grid = np.linspace(ev[0], budget, points)
    return grid, np.interp(grid, ev, fv)


def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="عدد مرات التكرار")
    ap.add_argument("--quick", action="store_true", help="تجربة سريعة")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs = 3 if args.quick else args.runs
    os.makedirs(args.out, exist_ok=True)

    rows, curves, finals, refs = [], {}, {}, {}
    for size, (n, m, budget) in SIZES.items():
        if args.quick:
            budget //= 5
        T = make_problem(n, m)
        g_fit = fitness(greedy(T)[None, :], T)[0]
        opt = brute_force(T) if size == "Small" else None
        refs[size] = (g_fit, opt)
        print(f"\n=== {size}: {n} tasks, {m} employees, budget={budget} evals "
              f"| Greedy={g_fit:.3f}" + (f" | Optimal={opt:.3f}" if opt else ""))
        for name, fn in ALGOS.items():
            vals, times, cs = [], [], []
            for r in range(runs): 
                rng = np.random.default_rng(1000 + r)
                t0 = time.perf_counter()
                _, bf, hist = fn(T, budget, rng)
                times.append(time.perf_counter() - t0)
                vals.append(bf)
                cs.append(to_curve(hist, budget)[1])
            grid = to_curve(hist, budget)[0]
            curves[(size, name)] = (grid, np.mean(cs, 0))
            finals[(size, name)] = vals
            rows.append([size, name, np.mean(vals), np.min(vals), np.max(vals),
                         np.std(vals), np.mean(times)])
            print(f"{name:7s} mean={np.mean(vals):8.3f} best={np.min(vals):8.3f} "
                  f"std={np.std(vals):6.3f} time={np.mean(times):6.2f}s")



    # --- حفظ الجداول ---
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Size", "Algorithm", "Mean", "Best", "Worst", "Std", "Time_s"])
        w.writerows([[r[0], r[1]] + [round(x, 4) for x in r[2:]] for r in rows])
    with open(os.path.join(args.out, "references.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Size", "Greedy", "Optimal(BruteForce)"])
        for s, (g, o) in refs.items():
            w.writerow([s, round(g, 4), "" if o is None else round(o, 4)])



    # --- منحنيات التقارب ---
    colors = {"GA": "tab:blue", "PSO": "tab:orange", "ACO": "tab:green", "Random": "gray"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, size in zip(axs, SIZES):
        for name in ALGOS:
            g, c = curves[(size, name)]
            ax.plot(g, c, label=name, color=colors[name])
        ax.axhline(refs[size][0], color="red", ls="--", lw=1, label="Greedy")
        if refs[size][1]:
            ax.axhline(refs[size][1], color="black", ls=":", lw=1, label="Optimal")
        ax.set_title(f"{size} (n={SIZES[size][0]}, m={SIZES[size][1]})")
        ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Best fitness (mean of runs)")
        ax.grid(alpha=.3)
        if size != "Small":
            lo = min(curves[(size, a)][1][-1] for a in ALGOS)
            ax.set_ylim(lo * 0.95, curves[(size, "GA")][1][5] * 1.05)
    axs[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "convergence.png"), dpi=150)
    plt.close()
    

    # --- استقرار النتائج (Boxplot) ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, size in zip(axs, SIZES):
        names = ["GA", "PSO", "ACO", "Random"]
        ax.boxplot([finals[(size, a)] for a in names], tick_labels=names)
        ax.axhline(refs[size][0], color="red", ls="--", lw=1)
        ax.set_title(f"{size}: final fitness over runs"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "boxplot.png"), dpi=150)
    plt.close()
    print(f"\nSaved results in ./{args.out}/")


if __name__ == "__main__":
    main()
