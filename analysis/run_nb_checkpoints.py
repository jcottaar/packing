import os
import time
import argparse
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Execute a Jupyter notebook cell-by-cell and periodically write an HTML "
            "snapshot (overwriting a single file) so you can inspect intermediate output."
        )
    )
    ap.add_argument("notebook", help="Path to .ipynb")
    ap.add_argument(
        "--outdir",
        default="nb_runs",
        help="Output directory (snapshot + final outputs).",
    )
    ap.add_argument(
        "--html-name",
        default="snapshot.html",
        help="HTML filename to overwrite within outdir.",
    )
    ap.add_argument(
        "--ipynb-name",
        default="snapshot.ipynb",
        help="Notebook filename to overwrite within outdir.",
    )
    ap.add_argument(
        "--every-seconds",
        type=float,
        default=120.0,
        help="Overwrite snapshot every N seconds.",
    )
    ap.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue execution even if a cell errors.",
    )
    ap.add_argument(
        "--kernel",
        default="",
        help="Kernel name (optional). If omitted, uses notebook kernelspec when available.",
    )

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    nb_path = os.path.abspath(args.notebook)
    nb_dir = os.path.dirname(nb_path) or "."

    nb = nbformat.read(nb_path, as_version=4)

    # Determine kernel name:
    # 1) explicit --kernel
    # 2) notebook metadata kernelspec.name
    # 3) let nbclient pick a default
    kernel_name = args.kernel.strip()
    if not kernel_name:
        kernel_name = (
            (nb.metadata.get("kernelspec") or {}).get("name")
            or ""
        )

    client_kwargs = dict(
        timeout=None,
        allow_errors=args.allow_errors,
        # Ensure relative paths inside the notebook behave like they do when run normally.
        resources={"metadata": {"path": nb_dir}},
    )
    if kernel_name:
        client_kwargs["kernel_name"] = kernel_name

    client = NotebookClient(nb, **client_kwargs)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    snap_ipynb = os.path.join(args.outdir, args.ipynb_name)
    snap_html = os.path.join(args.outdir, args.html_name)

    last_dump = time.time()
    start = last_dump

    # Simple lock so snapshot writing doesn't overlap with itself.
    # (We don't attempt to make notebook mutation thread-safe; nbclient only
    # mutates outputs when a cell finishes executing.)
    import threading
    snap_lock = threading.Lock()
    stop_event = threading.Event()

    def write_snapshot(tag: str):
        # Overwrite the same snapshot files each time.
        with snap_lock:
            nbformat.write(nb, snap_ipynb)
            body, _ = exporter.from_notebook_node(nb)
            tmp_html = snap_html + ".tmp"
            with open(tmp_html, "w", encoding="utf-8") as f:
                f.write(body)
            # Atomic-ish replace to avoid half-written HTML.
            os.replace(tmp_html, snap_html)
        now = time.time()
        print(f"[{tag} +{now-start:0.1f}s] wrote {snap_html}", flush=True)

    def snapshot_worker():
        # Writes a snapshot every N seconds, regardless of cell boundaries.
        # This ensures you get updates even if a single cell runs for a long time.
        next_t = time.time() + args.every_seconds
        while not stop_event.is_set():
            now = time.time()
            if now >= next_t:
                write_snapshot("checkpoint")
                next_t = now + args.every_seconds
            # Wake up frequently enough to be responsive but not wasteful.
            stop_event.wait(0.5)

    worker = threading.Thread(target=snapshot_worker, daemon=True)
    worker.start()

    # Write an initial snapshot so you can open the file immediately.
    write_snapshot("start")

    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            client.execute_cell(cell, i)

            # Also write a snapshot after each cell completes (useful when cells are short).
            write_snapshot(f"cell_{i:04d}")

        # Stop the background snapshotter and write a final snapshot.
    stop_event.set()
    worker.join(timeout=2.0)
    write_snapshot("done")


if __name__ == "__main__":
    main()
