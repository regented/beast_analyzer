Usage:

To generate a table with the data of all beasts:
~~~bash
python table.py
~~~
To do basic analysis on data (2000 simulations):
~~~bash
python analyze.py --data data.csv --default --html default.html
~~~
Further analysis can be done using the CLI commands described on analyze.py
~~~bash
python analyze.py --help
~~~
To run the web server and check the table (can be accessed at http://0.0.0.0:8000/default.html):
~~~bash
python -m http.server
~~~