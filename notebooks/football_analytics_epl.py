# Databricks notebook source
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/Volumes/workspace/raw/sidk_storage/E0.csv")

display(df)

# COMMAND ----------

from pyspark.sql.functions import *

home_stats = df.groupBy("HomeTeam").agg(
    count("*").alias("HomeMatches"),
    sum("FTHG").alias("GoalsForHome"),
    sum("FTAG").alias("GoalsAgainstHome")
).withColumnRenamed("HomeTeam","Team")

away_stats = df.groupBy("AwayTeam").agg(
    count("*").alias("AwayMatches"),
    sum("FTAG").alias("GoalsForAway"),
    sum("FTHG").alias("GoalsAgainstAway")
).withColumnRenamed("AwayTeam","Team")

team_stats = home_stats.join(
    away_stats,
    "Team",
    "outer"
).fillna(0)

team_stats = team_stats.withColumn(
    "Matches",
    col("HomeMatches")+col("AwayMatches")
).withColumn(
    "GoalsFor",
    col("GoalsForHome")+col("GoalsForAway")
).withColumn(
    "GoalsAgainst",
    col("GoalsAgainstHome")+col("GoalsAgainstAway")
).withColumn(
    "GoalDifference",
    col("GoalsFor")-col("GoalsAgainst")
)

display(team_stats)

# COMMAND ----------

home_points = df.withColumn(
    "Points",
    when(col("FTR")=="H",3)
    .when(col("FTR")=="D",1)
    .otherwise(0)
)

home_points = home_points.groupBy("HomeTeam")\
    .agg(sum("Points").alias("HomePoints"))\
    .withColumnRenamed("HomeTeam","Team")

away_points = df.withColumn(
    "Points",
    when(col("FTR")=="A",3)
    .when(col("FTR")=="D",1)
    .otherwise(0)
)

away_points = away_points.groupBy("AwayTeam")\
    .agg(sum("Points").alias("AwayPoints"))\
    .withColumnRenamed("AwayTeam","Team")

league_table = team_stats.join(home_points,"Team","left")\
                         .join(away_points,"Team","left")\
                         .fillna(0)

league_table = league_table.withColumn(
    "Points",
    col("HomePoints")+col("AwayPoints")
)

display(
    league_table.select(
        "Team",
        "Matches",
        "GoalsFor",
        "GoalsAgainst",
        "GoalDifference",
        "Points"
    ).orderBy(desc("Points"))
)

# COMMAND ----------

league_table.write.mode("overwrite").saveAsTable("epl_league_table")

# COMMAND ----------

from pyspark.sql.functions import round

league_table = league_table.withColumn(
    "WinPct",
    round((col("Points")/(col("Matches")*3))*100,2)
)

display(
    league_table.select(
        "Team",
        "Matches",
        "Points",
        "WinPct"
    ).orderBy(desc("WinPct"))
)

# COMMAND ----------

display(
    league_table.select(
        "Team",
        "GoalsFor",
        "GoalDifference",
        "Points"
    ).orderBy(desc("GoalsFor"))
)

# COMMAND ----------

home_wins = df.filter(col("FTR")=="H") \
    .groupBy("HomeTeam") \
    .count() \
    .withColumnRenamed("HomeTeam","Team") \
    .withColumnRenamed("count","HomeWins")
away_wins = df.filter(col("FTR")=="A") \
    .groupBy("AwayTeam") \
    .count() \
    .withColumnRenamed("AwayTeam","Team") \
    .withColumnRenamed("count","AwayWins")
performance = home_wins.join(
    away_wins,
    "Team",
    "outer"
).fillna(0)

display(performance)

# COMMAND ----------

matches = df.select(
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR"
)

display(matches)

# COMMAND ----------

matches.write.mode("overwrite").saveAsTable("epl_matches")

# COMMAND ----------

ranking = league_table.select(
    "Team",
    "Matches",
    "GoalsFor",
    "GoalsAgainst",
    "GoalDifference",
    "Points",
    "WinPct"
).orderBy(desc("Points"))

display(ranking)

# COMMAND ----------

ranking.write.mode("overwrite").saveAsTable("epl_team_ranking")

# COMMAND ----------

from pyspark.sql.functions import lit

home_results = df.select(
    col("Date"),
    col("HomeTeam").alias("Team"),
    when(col("FTR")=="H","W")
    .when(col("FTR")=="D","D")
    .otherwise("L")
    .alias("Result")
)

away_results = df.select(
    col("Date"),
    col("AwayTeam").alias("Team"),
    when(col("FTR")=="A","W")
    .when(col("FTR")=="D","D")
    .otherwise("L")
    .alias("Result")
)

form_table = home_results.union(away_results)

display(form_table)

# COMMAND ----------

form_table.write.mode("overwrite").saveAsTable("epl_team_form")

# COMMAND ----------

# Convert Spark dataframe to Pandas and sort by Points
ranking_pd = ranking.toPandas()
ranking_pd = ranking_pd.sort_values(by='Points', ascending=False)

# Display tabular data
display(ranking_pd)

import matplotlib.pyplot as plt

####################################
# 1. EPL Team Points
####################################
fig, ax = plt.subplots(figsize=(12, 6))

ranking_pd.set_index('Team')['Points'].plot(
    kind='bar',
    ax=ax,
    color='steelblue'
)

ax.set_title('EPL Team Points', fontsize=14)
ax.set_ylabel('Points')
ax.set_xlabel('Team')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

display(fig)
plt.close()


####################################
# 2. EPL Goals Scored
####################################
fig, ax = plt.subplots(figsize=(12, 6))

ranking_pd.set_index('Team')['GoalsFor'].plot(
    kind='bar',
    ax=ax,
    color='coral'
)

ax.set_title('EPL Goals Scored', fontsize=14)
ax.set_ylabel('Goals')
ax.set_xlabel('Team')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

display(fig)
plt.close()


####################################
# 3. Goal Difference
####################################
fig, ax = plt.subplots(figsize=(12, 6))

ranking_pd.set_index('Team')['GoalDifference'].plot(
    kind='bar',
    ax=ax,
    color='green'
)

ax.set_title('EPL Goal Difference', fontsize=14)
ax.set_ylabel('Goal Difference')
ax.set_xlabel('Team')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

display(fig)
plt.close()


####################################
# 4. Goals vs Points Bubble Chart
####################################
fig, ax = plt.subplots(figsize=(10,8))

# Bubble size based on absolute Goal Difference
bubble_size = ranking_pd['GoalDifference'].abs()*20 + 50

scatter = ax.scatter(
    ranking_pd['GoalsFor'],
    ranking_pd['Points'],
    s=bubble_size,
    c=ranking_pd['GoalDifference'],
    cmap='RdYlGn',
    alpha=0.8
)

# Add team labels
for i, team in enumerate(ranking_pd['Team']):
    ax.annotate(
        team,
        (
            ranking_pd['GoalsFor'].iloc[i],
            ranking_pd['Points'].iloc[i]
        ),
        fontsize=8
    )

plt.colorbar(scatter, label='Goal Difference')

ax.set_xlabel('Goals Scored')
ax.set_ylabel('Points')
ax.set_title('Goals vs Points (Bubble Size = Goal Difference)')

plt.tight_layout()

display(fig)
plt.close()

# COMMAND ----------

