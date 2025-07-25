# Simulated Annealing for Rectangle Packing

This project implements a **Simulated Annealing (SA)** algorithm to optimally arrange rectangular crates inside a fixed warehouse grid. The objective is to minimize the perimeter of the smallest bounding rectangle that contains all crates, while ensuring **no overlaps** and **no rotation** of crates.

---

 Problem Statement

Given:
- A warehouse of size `W × H`
- A set of rectangular crates with integer dimensions `(width, height)`

**Goal**:  
Place all crates inside the warehouse:
- Without overlapping
- Without rotation
- While minimizing the perimeter of the bounding rectangle that contains all crates.

---

Algorithm Overview

The algorithm is based on **Simulated Annealing (SA)**:
1. Start with a **random initial placement** of crates.
2. Generate **neighbor states**:
   - Randomly move a crate within the warehouse.
   - Randomly swap the positions of two crates.
3. Evaluate the perimeter of the bounding box.
4. Decide whether to accept the new state using the **Metropolis acceptance rule**:
   \[
   P = \exp\left(-\frac{\Delta}{T}\right)
   \]
   where `Δ` is the change in perimeter and `T` is the current temperature.
5. "Cool down" the temperature according to a logarithmic schedule.
6. Stop when the target perimeter is reached or when no improvements are found.


