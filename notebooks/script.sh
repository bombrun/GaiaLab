jupyter nbconvert --to python "$1" --stdout > notebook.py \
&& (python notebook.py; rm notebook.py)
