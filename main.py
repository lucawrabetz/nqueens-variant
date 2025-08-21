import time
import gurobipy as gp
import numpy as np
from gurobipy import GRB

_DEBUG = False
_TIME_LIMIT_S = 120


def dprint(msg) -> None:
    if _DEBUG:
        print(msg)


def get_n() -> int:
    n_dim: int = 0
    while n_dim == 0:
        try:
            n_dim = int(input("N: "))
            dprint(f"n_dim: {n_dim}")

        except ValueError as _:
            print("\nPlease input an integer for N.\n")

    return n_dim


class NQueensBIP:
    def __init__(self, n_dim: int, disjunctive=True):
        self.n_dim: int = n_dim
        self.model = gp.Model("NQueensBIP")
        self.model.setParam("TimeLimit", _TIME_LIMIT_S)
        output_flag: int = 1 if _DEBUG else 0
        self.model.setParam("OutputFlag", output_flag)
        self.x_vars: list[list] = [[] for _ in range(n_dim)]
        self.y_vars: list[list] = [[] for _ in range(n_dim)]
        self._disjunctive: bool = disjunctive

    def configure_bip(self) -> None:
        dprint(self.model)

        self._add_variables()
        dprint(f"vars: \nx: {self.x_vars},\ny: {self.y_vars}")

        if self._disjunctive:
            self._add_constraints_disjunctive()
        else:
            self._add_constraints()
        dprint(f"constraints: {self.model.getConstrs()}")

        self._add_objective()

    def solve(self) -> None:
        tic: float = time.time()
        self.model.optimize()
        toc: float = time.time()

        runtime_ms: float = 1000 * (toc - tic)
        print(
            f"\nSolved NQueens BIP (N = {self.n_dim}) for {runtime_ms} ms, objective: {int(self.model.ObjVal)} attackable cells out of {int(self.n_dim * self.n_dim)}."
        )
        if self.model.getAttr("Status") == GRB.OPTIMAL:
            print("Solution is optimal.")
        elif self.model.getAttr("Status") == GRB.TIME_LIMIT:
            print(f"Optimality gap: {self.model.getAttr('MIPGap')}")
        else:
            print("No solution found.")

    def print_solution(self) -> None:
        print("\nPlacement (0 = queen placed):")
        self._print_grid(self.x_vars)
        print("\nAttackable (0 = safe, X = attackable):")
        self._print_grid(self.y_vars, "0", "X")

    def _print_grid(self, binaries: list, default: str = "X", one: str = "0") -> None:
        for i in range(self.n_dim):
            row_list: list[str] = [default for _ in range(self.n_dim)]
            for j in range(self.n_dim):
                if binaries[i][j].X > 0.5:
                    row_list[j] = one
            print(f"{'  '.join(row_list)}")

    def _add_variables(self) -> None:
        dprint("Adding vars...")
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                self.x_vars[i].append(
                    self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}")
                )
                self.y_vars[i].append(
                    self.model.addVar(vtype=GRB.BINARY, name=f"y_{i}{j}")
                )
        self.model.update()

    def _add_constraints(self) -> None:
        dprint("Adding constraints...")
        self._add_place_n_queens_constraint()
        self._add_rc_attack_constraints()
        self._add_diag_attack_constraints()

    def _add_constraints_disjunctive(self) -> None:
        dprint("Adding disjunctive constraints...")
        self._add_place_n_queens_constraint()
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                # initialize the attacker set with the cell (i, j) itself.
                attacker_set: list[tuple[int, int]] = [(i, j)]

                # add the entire row to the attacker set.
                row_set: list[tuple[int, int]] = [
                    (i, l) for l in range(self.n_dim) if l != j
                ]
                attacker_set.extend(row_set)

                # add the entire column to the attacker set.
                column_set: list[tuple[int, int]] = [
                    (k, j) for k in range(self.n_dim) if k != i
                ]
                attacker_set.extend(column_set)

                # add diagonals to the attacker set.
                for k in range(self.n_dim):
                    for l in range(self.n_dim):
                        if i == k or j == l:
                            continue
                        if k - i == l - j or k - i == j - l:
                            attacker_set.append((k, l))

                big_m = len(attacker_set)
                self.model.addConstr(
                    big_m * self.y_vars[i][j]
                    >= gp.quicksum(self.x_vars[k][l] for (k, l) in attacker_set)
                )

    def _add_place_n_queens_constraint(self) -> None:
        dprint("Adding place n queens constraint...")
        self.model.addConstr(
            gp.quicksum(
                self.x_vars[i][j] for i in range(self.n_dim) for j in range(self.n_dim)
            )
            == self.n_dim,
            name="c_place_n_queens",
        )
        self.model.update()

    def _add_rc_attack_constraints(self) -> None:
        dprint("Adding row & column attack constraints...")
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                for k in range(self.n_dim):
                    self.model.addConstr(self.y_vars[i][j] >= self.x_vars[i][k])
                    self.model.addConstr(self.y_vars[i][j] >= self.x_vars[k][j])

        self.model.update()

    def _add_diag_attack_constraints(self) -> None:
        dprint("Adding diagonal attack constraints...")
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                for k in range(self.n_dim):
                    for l in range(self.n_dim):
                        if k - i == l - j:
                            self.model.addConstr(self.y_vars[i][j] >= self.x_vars[k][l])
                        if k - i == j - l:
                            self.model.addConstr(self.y_vars[i][j] >= self.x_vars[k][l])

        self.model.update()

    def _add_objective(self) -> None:
        self.model.setObjective(
            gp.quicksum(
                self.y_vars[i][j] for i in range(self.n_dim) for j in range(self.n_dim)
            ),
            GRB.MINIMIZE,
        )
        self.model.update()


def main() -> None:
    n_dim: int = get_n()
    nqueens_bip = NQueensBIP(n_dim, disjunctive=False)
    nqueens_bip.configure_bip()
    nqueens_bip.solve()
    nqueens_bip.print_solution()


if __name__ == "__main__":
    main()
