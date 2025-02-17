# Greedy Algorithm Problems

## Overview
This repository contains solutions to various problems solved using the Greedy Algorithm approach in Python. Each problem demonstrates an optimal way to solve a given challenge efficiently.

## Problems Solved

### 1. Minimum Coins to Make Change
**Problem:** Find the minimum number of coins needed to make a given sum.

**Approach:**
- Uses recursion with memoization (top-down approach) to minimize redundant calculations.
- Implements a decision-making process where a coin is either taken or not taken to reach the target sum.
- Returns the minimum number of coins required.

**Code:** [See `minCoins.py`](minCoins.py)

---

### 2. Minimum Platforms Required
**Problem:** Find the minimum number of platforms needed at a railway station so that no train waits.

**Approach:**
- Sorts the arrival and departure times separately.
- Uses a two-pointer approach to track overlapping train timings.
- Maintains a count of required platforms and updates the maximum needed.

**Code:** [See `minimumPlatforms.py`](minimumPlatforms.py)

---

### 3. Maximum Meetings in a Room
**Problem:** Schedule the maximum number of meetings in a single conference room.

**Approach:**
- Sorts meetings by their ending time to ensure optimal usage of available time slots.
- Iterates through sorted meetings and selects non-overlapping ones.

**Code:** [See `maximumMeetings.py`](maximumMeetings.py)

---

### 4. Assign Cookies to Children
**Problem:** Given greed factors of children and sizes of cookies, find the maximum number of content children.

**Approach:**
- Sorts both greed factors and cookie sizes.
- Uses a two-pointer approach to assign cookies to children in an optimal way.

**Code:** [See `findContentChildren.py`](findContentChildren.py)

## How to Run
1. Clone the repository.
2. Ensure Python is installed on your system.
3. Run the respective Python file using:
   ```bash
   python filename.py
   ```

## Contributing
Feel free to contribute by adding more problems, optimizing existing solutions, or improving documentation.

## License
This repository is licensed under the MIT License.