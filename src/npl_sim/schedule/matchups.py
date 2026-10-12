"""How many games each pair plays, and where, as a list of series.

MLB's balanced mix, per team (162):
- division 52: 18 vs its rival (9 home), 17 vs the other two (9 home vs one,
  8 vs the other, swapping each season)
- own league 64: 8 vs each of 8 teams, 5 in one park (3 + 2) and 3 in the
  other; each team hosts 5 vs four of them and 3 vs the other four, swapping
  each season
- interleague 46: 4 vs 10 teams as a 2-game series in each park, and 3 vs 2
  teams as one series in one park. The 3-game opponents rotate each season,
  and each 3-game pair swaps parks from one meeting to the next.
Every team ends up with exactly 81 home games, all in 2- and 3-game series
(Dave, 2026-10-12: MLB-style 2- or 3-game series with off and travel days).
"""

from __future__ import annotations

from dataclasses import dataclass

from npl_sim.league import Alignment


@dataclass(frozen=True)
class Series:
    home: str
    away: str
    games: int
    kind: str  # rival | division | league | interleague

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.home, self.away))


def split_games(n: int) -> list[int]:
    """Split one side's games against an opponent into 3- and 2-game series."""
    if n <= 3:
        return [n]
    twos = {0: 0, 1: 2, 2: 1}[n % 3]
    return [3] * ((n - 2 * twos) // 3) + [2] * twos


def _series(home: str, away: str, games: int, kind: str) -> list[Series]:
    return [Series(home, away, g, kind) for g in split_games(games)]


def interleague_offsets(season: int) -> tuple[int, int]:
    """Offsets of the two 3-game interleague matchings; they rotate yearly."""
    k = (2 * season) % 12
    return k, k + 1


def build_series(alignment: Alignment, season: int) -> list[Series]:
    flip = season % 2 == 1
    out: list[Series] = []

    for league in alignment.leagues:
        for div in league.divisions:
            (a, b), (c, d) = div.rival_pairs()
            for x, y in ((a, b), (c, d)):
                out += _series(x, y, 9, "rival") + _series(y, x, 9, "rival")
            # Non-rival pairs form the 4-cycle a-c-b-d-a; walking it gives each
            # team one 9-home and one 8-home opponent.
            for x, y in ((a, c), (c, b), (b, d), (d, a)):
                if flip:
                    x, y = y, x
                out += _series(x, y, 9, "division") + _series(y, x, 8, "division")

        # Own-league pairs: the teams of two divisions form a 4x4 grid; the
        # team whose indices sum to an even number hosts 5, so everyone hosts
        # 5 vs two of each other division's teams. Flips each season.
        divs = league.divisions
        for a in range(len(divs)):
            for b in range(a + 1, len(divs)):
                for i, x in enumerate(divs[a].teams):
                    for j, y in enumerate(divs[b].teams):
                        five, three = (y, x) if (i + j + season) % 2 else (x, y)
                        out += _series(five, three, 5, "league")
                        out += _series(three, five, 3, "league")

    al, nl = ([t for div in lg.divisions for t in div.teams] for lg in alignment.leagues)
    n = len(al)
    three = interleague_offsets(season)
    for i, x in enumerate(al):
        for k in range(n):
            y = nl[(i + k) % n]
            if k not in three:
                out += _series(x, y, 2, "interleague") + _series(y, x, 2, "interleague")
                continue
            # The two 3-game offsets differ in parity, so each team hosts one.
            home, away = (x, y) if (k + season) % 2 == 0 else (y, x)
            out.append(Series(home, away, 3, "interleague"))
    return out
