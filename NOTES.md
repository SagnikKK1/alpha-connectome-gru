# Measurement notes

The model was rarely the bottleneck. Knowing whether a change was real was.

**Seed noise is bigger than most effects.** Two runs that differ only in seed differ by about 0.0015 on an 800-sequence
screen. Almost every idea I tested had a true effect smaller than that. A one-seed comparison of single models tells you
nothing.

**Judge at the ensemble size you ship.** An ensemble of n members scores roughly `A - c/n`, where `A` is the quality of the
recipe and `c` is member noise. A change that only lowers `c` looks great when you compare pairs, and is worth almost
nothing once eight members already average the noise away. I fit `A` and `c` on 1-, 2- and 4-member subsets and read the
delta at n = 8.

**Searching on your own screen overpredicts.** When I picked the best of many configurations on the screen, the gain I
then saw on the leaderboard was about a fifth of the screen gain. Only a nested split (select on one half, measure on the
other) predicted the leaderboard.

**Hold out by series, not by row.** The sequences are chunks of longer continuous streams. A held-out chunk from the same
stream as a training chunk is easier than a truly unseen one. The hidden test set shares no streams with the training data.

**Not every family transfers.** The data contains four instrument families. Gains on one family (the hardest, with almost
half the metric weight) showed up on my screen and never on the leaderboard. Only gains on the other three transferred.

**Output tricks fitted on validation do not survive.** Per-family scaling and power transforms gained offline and lost on
the leaderboard every time.

**Public and private disagree.** We finished 47th on the public board and 57th on the private one.
