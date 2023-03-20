from setuptools import setup

setup(
    name='buildstock_query',
    version='2022.10.8',
    description="Python library for querying and analyzing ResStock and ComStock",
    author="Rajendra Adhikari",
    author_email='Rajendra.Adhikari@nrel.gov',
    packages=['buildstock_query'],
    python_requires='>=3.9',
    install_requires=[
        "pandas >= 1.5.0",
        "pyarrow >= 9.0.0",
        "s3fs[boto3] >= 2022.8.2",
        "pyathena == 2.13.0",
        "SQLAlchemy == 1.4.46",
        "dask >= 2022.9.2",
        "colorama >= 0.4.5",
        "inquirerpy >= 0.3.4",
        "types-PyYAML >= 6.0.12.2"
    ],

    extra_require={
        'dev': ["pytest >= 7.1.3",
                "flake8 >= 5.0.4",
                "pdoc3 >= 0.10.0",
                "autopep8 >= 1.7.0",
                "dash-bootstrap-components >= 1.2.1",
                "dash-extensions >= 0.1.6",
                "dash-mantine-components >= 0.10.2",
                "dash-iconify >= 0.1.2",
                "coverage >= 6.5.0",
                "plotly >= 5.10.0",
                "dash >= 2.6.2"],
        'full': ["dash-bootstrap-components >= 1.2.1",
                 "dash-extensions >= 0.1.6",
                 "dash-mantine-components >= 0.10.2",
                 "dash-iconify >= 0.1.2",
                 "plotly >= 5.10.0",
                 "dash >= 2.6.2"]

    },
    entry_points={
        'console_scripts': ['upgrades_analyzer=buildstock_query.tools.upgrades_analyzer:main',
                            'upgrades_visualizer=buildstock_query.tools.upgrades_visualizer:main']
    },
    
)
