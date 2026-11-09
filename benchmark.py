#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynXS benchmark
===============
Measures, for a set of 2D detector images and a pyFAI geometry file, the per-frame
cost of the four steps that the Integration Tool performs:

    read  ->  azimuthal integration  ->  write ASCII curve  ->  (optional) PNG preview

The numbers printed here are the ones reported in Sect. 3.2 of the manuscript.

Usage
-----
    python benchmark.py --images "D:\\data\\*.cbf" --poni geometry.poni --bins 2000 --repeat 3

    optional:  --unit q_A^-1   --no-png   --out results.csv
"""
import argparse
import glob
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Per-frame benchmark of the SynXS integration path")
    ap.add_argument("--images", required=True, help="glob pattern of the 2D images")
    ap.add_argument("--poni", required=True, help="pyFAI geometry file")
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--unit", default="q_A^-1")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--out", default=None, help="optional CSV output")
    args = ap.parse_args()

    import tempfile

    import fabio
    import pyFAI

    pyFAI.use_opencl = False
    os.environ["PYFAI_OPENCL"] = "False"

    files = sorted(glob.glob(args.images))
    if not files:
        raise SystemExit("no images matched %s" % args.images)

    tmpdir = tempfile.mkdtemp(prefix="synxs_bench_")
    ai = pyFAI.load(args.poni)
    print("geometry : %s" % args.poni)
    print("detector : %s" % ai.detector.name)
    print("distance : %.4f m   wavelength : %.4f A" % (ai.dist, ai.wavelength * 1e10))
    print("images   : %d   bins: %d   repeat: %d\n" % (len(files), args.bins, args.repeat))

    rows = []
    for path in files:
        t = {}
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            img = fabio.open(path).data.astype(np.float32)
            t["read"] = min(t.get("read", 1e9), time.perf_counter() - t0)

            t0 = time.perf_counter()
            q, I = ai.integrate1d(img, args.bins, unit=args.unit)
            t["integrate"] = min(t.get("integrate", 1e9), time.perf_counter() - t0)

            t0 = time.perf_counter()
            np.savetxt(os.path.join(tmpdir, "curve.dat"), np.column_stack((q, I)))
            t["write"] = min(t.get("write", 1e9), time.perf_counter() - t0)

            if not args.no_png:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                t0 = time.perf_counter()
                fig = plt.figure(figsize=(6, 4))
                plt.plot(q, I)
                plt.tight_layout()
                fig.savefig(os.path.join(tmpdir, "curve.png"), dpi=150)
                plt.close(fig)
                t["png"] = min(t.get("png", 1e9), time.perf_counter() - t0)
        rows.append((os.path.basename(path), t))

    keys = ["read", "integrate", "write"] + ([] if args.no_png else ["png"])
    print("%-34s" % "image" + "".join("%12s" % k for k in keys) + "%12s" % "total")
    for name, t in rows:
        total = sum(t[k] for k in keys)
        print("%-34s" % name + "".join("%10.1f ms" % (t[k] * 1e3) for k in keys)
              + "%10.1f ms" % (total * 1e3))

    means = {k: np.mean([t[k] for _, t in rows]) for k in keys}
    print("-" * (34 + 12 * (len(keys) + 1)))
    print("%-34s" % "mean" + "".join("%10.1f ms" % (means[k] * 1e3) for k in keys)
          + "%10.1f ms" % (sum(means.values()) * 1e3))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("image," + ",".join(keys) + ",total_ms\n")
            for name, t in rows:
                fh.write(name + "," + ",".join("%.3f" % (t[k] * 1e3) for k in keys)
                         + ",%.3f\n" % (sum(t[k] for k in keys) * 1e3))
        print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
