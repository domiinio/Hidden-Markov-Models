# Hidden-Markov-Models
This repository contains work solving the problem of estimating robot's position using HMM.

## Project Description
In this repository, I present the solution to a second semestral project in the [Artificial Intelligence](https://moodle.fel.cvut.cz/local/kos/pages/course/info.php?code=B3M33UI&semester=B162) course, at CTU.
The goal of this project was to use Hidden Markov Models to estimate position of a mobile robot that moves in a maze. I did this project together with Tomáš Rutrle and here I present our joint results.

We used a couple of HMM-based prediction algorithms, namely *filtering*, *smoothing*, and the *Vitterbi* algorithm. We also compared different implementations of mentioned algorithms, such as using log 
probabilities with Vitterbi for better numerical stability. Finally, we have implemented the *Baum-Welch* algorithm for estimating HMM parameters.

To validate the performance of implemented algorithms we used the Manhattan distance of the estimated and real positions as well as a simple hit/miss metric.

## Documentation
Detailed documentation and description of the implemented algorithms and solution are in the pdf report that can be found in the `report` folder accessible from the root of this repository. 

## Results
Here, I visualise some of the key achieved results. Description, discussion and comments on the visualised figures can be found in the report. 
