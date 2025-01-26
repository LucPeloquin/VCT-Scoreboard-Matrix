import csv
from collections import defaultdict

# Input and output file paths
input_csv = 'filtered_text.csv'
output_csv = 'player_stats.csv'

# Initialize counters for kills and deaths
kills_counter = defaultdict(int)
deaths_counter = defaultdict(int)

# Read the filtered CSV and update counters
with open(input_csv, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row

    for row in reader:
        round_number, detected_time, player, player2 = row

        # Update kills counter for "Player"
        if player:
            kills_counter[player] += 1

        # Update deaths counter for "Player 2"
        if player2:
            deaths_counter[player2] += 1

# Write the player stats to a new CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Player", "Kills", "Deaths"])  # Header

    # Combine all unique players from both counters
    all_players = set(kills_counter.keys()).union(set(deaths_counter.keys()))

    # Write stats for each player
    for player in sorted(all_players):
        kills = kills_counter.get(player, 0)
        deaths = deaths_counter.get(player, 0)
        writer.writerow([player, kills, deaths])

print(f"Player stats saved to {output_csv}")