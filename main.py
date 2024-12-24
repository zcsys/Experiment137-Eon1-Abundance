from simulation import Simulation
from things import Things
import argparse

POP_0 = 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type = str)
    args = parser.parse_args()

    if args.load:
        things_instance = Things(state_file = args.load)
    else:
        things_instance = Things(["monad"] * POP_0)

    simulation = Simulation(things_instance, load_file = args.load)
    simulation.run()
