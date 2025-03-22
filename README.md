# BlueROV2 Tether-Enhanced Model

This repository provides a Python-based BlueROV2 (heavy configuration) dynamics model, extended to optionally include a lumped-mass tether. The code references:

- von Benzon, M.; Sørensen, F.F.; Uth, E.; Jouffroy, J.; Liniger, J.; Pedersen, S.  
  *An Open-Source Benchmark Simulator: Control of a BlueROV2 Underwater Robot.*  
  J. Mar. Sci. Eng. 2022, 10, 1898.
- T.I. Fossen. *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed. Wiley, 2021.

---

## Files

1. **BlueROV2.py**  
   Contains the core `BlueROV2` class for 6-DOF ROV dynamics, plus an optional `Tether` class for lumped-mass tether modeling.  
   - To activate the tether:  
     ```python
     rov.use_tether = True
     rov.tether = Tether(...)
     rov.tether_state = <initial array>  
     rov.anchor_pos = [x0, y0, z0]
     ```
   - On each call to `rov.dynamics(...)`, tension forces are added in body-frame.

2. **test_euler.py**  
   Demonstrates **simple forward-Euler** integration of the ROV (and tether if enabled). This prints results at each step in real time.

3. **test_ode.py**  
   Demonstrates **implicit integration** using `scipy.integrate.solve_ivp(method="BDF")`. This is more stable for stiff tether dynamics. Prints results at each sampled time.

Both the Euler and ODE methods should yield consistent results (though the implicit solver is typically more stable for large thrust or stiff tethers).

---

## Installation

1. Clone or download this repository.
2. Install dependencies:
    ```bash
    pip install numpy scipy
    ```
3. Run:
    ```
    python test_euler.py
    ```
    or
    ```
    python test_ode.py
    ```

---

## License

This project is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute.
Contributions are welcome!