# Hidden-Markov-Models
This repository contains work solving the problem of estimating robot's position using HMM.

## Project Description
In this repository, I present the solution to a second semestral project in the [Artificial Intelligence](https://moodle.fel.cvut.cz/local/kos/pages/course/info.php?code=B3M33UI&semester=B162) course, at CTU.
The goal of this project was to use Hidden Markov Models to estimate position of a mobile robot that moves in a maze. I did this project together with Tomáš Rutrle and here I present our joint results.

![Example setting of a robot's position in a map](figs/maze.png)

In the figure above, there is a visualisation of the problem setting. We try to localise the robot (yellow dot) in a map and guess on which tile it currently is, given an uncertain transition model and the 
sequence of commands.

We used a couple of HMM-based prediction algorithms, namely *filtering*, *smoothing*, and the *Vitterbi* algorithm. We also compared different implementations of mentioned algorithms, such as using log 
probabilities with Vitterbi for better numerical stability. Finally, we have implemented the *Baum-Welch* algorithm for estimating HMM parameters.

To validate the performance of implemented algorithms we used the Manhattan distance of the estimated and real positions as well as a simple hit/miss metric.

## Repository Structure

  - `code` folder contains Python scripts that solve the afore-described task. It also contains `Matlab` script that creates some of the plots used to visualise the results.
  - `figs` folder contains figures used either in this README or in the report
  - `mazes` folder contains files that describe used environments/maps where the robot was moving.
  - `report` folder contains a pdf report presenting the work done and detailing the implemented algorithms and results.

## Documentation
Detailed documentation and description of the implemented algorithms and solution are in the pdf report that can be found in the `report` folder accessible from the root of this repository. 

## Results
Here, I visualise some of the key achieved results. Description, discussion and comments on the visualised figures can be found in the report. 

### Hit Rate
![Comparison of hit rates of individual algorithms](figs/hit_rate_cmp.png)

### Error of the Position Estimator
![Comparison of the position error](figs/error_cmp.png)
