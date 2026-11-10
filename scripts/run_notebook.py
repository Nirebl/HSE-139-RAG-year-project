"""Execute the main hand-authored EDA.ipynb in a clean kernel (does not regenerate it)."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
path=root/'EDA.ipynb'
notebook=nbformat.read(path,as_version=4)
NotebookClient(notebook,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(notebook,path)
print('Successfully executed:',path)
