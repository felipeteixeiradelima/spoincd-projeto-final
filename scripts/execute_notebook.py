"""
Executor programático de notebooks Jupyter para preenchimento de saídas e gráficos.
"""

import base64
import contextlib
import io
import json
from pathlib import Path
import sys

from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def execute_notebook(nb_path: Path):
    print(f"Executando notebook: {nb_path.name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    # Mudar diretório para a pasta do notebook para compatibilidade de caminhos relativos
    old_cwd = Path.cwd()
    notebook_dir = nb_path.parent

    try:
        import os
        os.chdir(notebook_dir)

        execution_count = 0
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue

            execution_count += 1
            cell["execution_count"] = execution_count
            code = "".join(cell["source"])

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            plt.close("all")

            print(f"  -> Executando célula de código {execution_count}...")
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = shell.run_cell(code)

            cell["outputs"] = []

            # Captura stdout
            out_str = stdout_buf.getvalue()
            if out_str:
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_str.splitlines(keepends=True),
                })

            # Captura stderr
            err_str = stderr_buf.getvalue()
            if err_str:
                cell["outputs"].append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": err_str.splitlines(keepends=True),
                })

            # Captura gráficos matplotlib
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig in figs:
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
                cell["outputs"].append({
                    "data": {
                        "image/png": img_base64,
                        "text/plain": ["<Figure size ...>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                })
            plt.close("all")

            # Erros de execução
            if result.error_in_exec:
                err = result.error_in_exec
                print(f"ERRO na célula {execution_count}: {err}")
                cell["outputs"].append({
                    "ename": type(err).__name__,
                    "evalue": str(err),
                    "output_type": "error",
                    "traceback": [str(err)],
                })
                break

    finally:
        os.chdir(old_cwd)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Notebook {nb_path.name} executado e salvo com sucesso com todas as saídas e figuras!")


if __name__ == "__main__":
    target_nb = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "notebooks" / "03-modelagem-risco.ipynb")
    execute_notebook(Path(target_nb))
