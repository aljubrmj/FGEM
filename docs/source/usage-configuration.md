# Configuration

The FGEM workflow requires the user to input a configuration file to describe the project components, i.e., metadata, finanical parameters, subsurface reservoir, wellbore, weather, power plant, energy storage (thermal and/or electrochemical), and power markets (wholesale, capacity, green credits, and/or PPA). For a given run, only the relevant components are required to be configured. FGEM validates the input configuration and sets default paramters where paramters are missing. FGEM then simulates the project lifetime, stores records, computes economics, and visualizes data. Herein, we describe the main configuration paramters. 

## `metadata`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/metadata.csv
```

## `subsurface`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/subsurface.csv
```

## `power plant`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/powerplant.csv
```

## `market/weather`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/market_weather.csv
```

## `battery`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/battery.csv
```

## `thermal energy storage tank`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,10,10,33,10
   :file: _static/tank.csv
```


