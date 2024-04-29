# Configuration

The FGEM workflow requires the user to input a configuration file to describe the project components, i.e., metadata, finanical parameters, subsurface reservoir, wellbore, weather, power plant, energy storage (thermal and/or electrochemical), and power markets (wholesale, capacity, green credits, and/or PPA). For a given run, only the relevant components are required to be configured. FGEM validates the input configuration and sets default paramters where paramters are missing. FGEM then simulates the project lifetime, stores records, computes economics, and visualizes data. Herein, we describe the main configuration paramters. 

## `metadata`
