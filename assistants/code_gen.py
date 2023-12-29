
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'path_to_your_file.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'Meetup Date' to datetime for proper plotting
data['Meetup Date'] = pd.to_datetime(data['Meetup Date'])

# Sort data by date for proper plotting in line chart
data.sort_values(by='Meetup Date', inplace=True)

# Plot the membership growth over time as a linear chart
plt.figure(figsize=(10, 6))
plt.plot(data['Meetup Date'], data['Members'], marker='o', linestyle='-', color='b')
plt.title('Ray Meetup Membership Growth Over Years')
plt.xlabel('Meetup Date')
plt.ylabel('Members')
plt.grid(True)
plt.tight_layout()

# Save the line chart
plt.savefig('ray_growth_meetup.png')
plt.close()

# Now let's create the stacked bar charts for RSVPs and Attended
# First, fill NaN Attended values with zeros
data['Attended'].fillna(0, inplace=True)

# Plotting the stacked bar charts
plt.figure(figsize=(14, 8))
bar_width = 0.4
r1 = range(len(data['Meetup Date']))
r2 = [x + bar_width for x in r1]

plt.bar(r1, data['RSVP'], color='c', width=bar_width, edgecolor='grey', label='RSVP')
plt.bar(r2, data['Attended'], color='m', width=bar_width, edgecolor='grey', label='Attended')
plt.xlabel('Meetup Date', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(data['Meetup Date']))], data['Meetup Date'].dt.strftime('%m/%d/%Y'), rotation=30)
plt.title('RSVPs and Attended per Meetup Date')
plt.legend()

# Save the bar charts
plt.savefig('rsvp_attended.png')
plt.close()
