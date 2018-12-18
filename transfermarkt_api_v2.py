import requests
from bs4 import BeautifulSoup
import pandas as pd

TRANSFER_MARKT_URL = 'https://www.transfermarkt.co.uk'
TRANSFER_MARKT_BROWSE_PATH = 'premier-league/gesamtspielplan/wettbewerb/GB1?saison_id={season}&spieltagVon=1&spieltagBis=38'

HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

TEAM_SHORT_NAME_LOOKUP = {
    'Arsenal': 'ARS',
    'Aston Villa': 'AVL',
    'Birmingham': 'BIR',
    'Blackburn': 'BLB',
    'Blackpool': 'BLA',
    'Bolton': 'BOL',
    'Bournemouth': 'BOU',
    'Burnley': 'BUR',
    'Cardiff': 'CAR',
    'Chelsea': 'CHE',
    'Crystal Palace': 'CRY',
    'Everton': 'EVE',
    'Fulham': 'FUL',
    'Hull City': 'HUL',
    'Leicester': 'LEI',
    'Liverpool': 'LIV',
    'Man City': 'MCI',
    'Man Utd': 'MUN',
    'Middlesbrough': 'MID',
    'Newcastle': 'NEW',
    'Norwich': 'NOR',
    'Portsmouth': 'POR',
    'QPR': 'QPR',
    'Reading': 'REA',
    'Southampton': 'SOU',
    'Stoke City': 'STK',
    'Sunderland': 'SUN',
    'Swansea': 'SWA',
    'Spurs': 'TOT',
    'Watford': 'WAT',
    'West Brom': 'WBA',
    'West Ham': 'WHU',
    'Wigan': 'WIG',
    'Wolves': 'WOL'
}


def _extract_matchday_team_ranks(season, league_id):
    page = '{url}/{path}'.format(url=TRANSFER_MARKT_URL,
                                 path=TRANSFER_MARKT_BROWSE_PATH.format(season=season.split('/')[0]))
    page_tree = requests.get(page, headers=HEADERS)
    page_soup = BeautifulSoup(page_tree.content, 'html.parser')

    # print(page_soup.prettify())

    matchday_tables = page_soup.select('div.table-header ~ table')
    gameweeks = [[matches.find_all('td') for matches in matchday.find_all('tr')
                  if len(matches.find_all('td')) >= 5] for matchday in matchday_tables]
    # print(len(gameweeks))
    # print([len(matches) for matches in gameweeks])

    list_matches = []
    running_stats = {'home': {}, 'away': {}}

    for index, matches in enumerate(gameweeks):
        stage = index + 1
        date = None
        for match in matches:
            # print([entry.text for entry in match])
            date_str_unformatted = match[0].text.strip()
            date_str = ' '.join(date_str_unformatted.split()) if date_str_unformatted != '' else None
            if date_str is not None:
                date = date_str

            home = match[2].text.split(')')
            away = match[6].text.split('(')

            home_short_name = TEAM_SHORT_NAME_LOOKUP.get(home[1].strip(), home[1].strip())
            if home_short_name == home[1].strip() and home_short_name not in ['QPR']:
                print('Short name not found', home_short_name)
            if home_short_name not in running_stats['home']:
                running_stats['home'][home_short_name] = dict({
                    'goals_scored': [],
                    'goals_conceded': [],
                    'goals_scored_total': 0,
                    'goals_conceded_total': 0,
                    'matches_played': 0
                })

            away_short_name = TEAM_SHORT_NAME_LOOKUP.get(away[0].strip(), away[0].strip())
            if away_short_name == away[0].strip() and away_short_name not in ['QPR']:
                print('Short name not found', away_short_name)
            if away_short_name not in running_stats['away']:
                running_stats['away'][away_short_name] = dict({
                    'goals_scored': [],
                    'goals_conceded': [],
                    'goals_scored_total': 0,
                    'goals_conceded_total': 0,
                    'matches_played': 0
                })

            scores = match[4].text.split(':')
            home_goals = int(scores[0].strip())
            away_goals = int(scores[1].strip())

            home_goals_scored = pd.Series(running_stats['home'][home_short_name]['goals_scored'])
            home_goals_conceded = pd.Series(running_stats['home'][home_short_name]['goals_conceded'])

            away_goals_scored = pd.Series(running_stats['away'][away_short_name]['goals_scored'])
            away_goals_conceded = pd.Series(running_stats['away'][away_short_name]['goals_conceded'])

            this_match = [
                stage,
                date,
                home_short_name,
                int(home[0][1:-1]),
                running_stats['home'][home_short_name]['matches_played'],
                running_stats['home'][home_short_name]['goals_scored_total'],
                running_stats['home'][home_short_name]['goals_conceded_total'],
                home_goals_scored.min(),
                home_goals_scored.max(),
                home_goals_scored.mean(),
                home_goals_scored.median(),
                home_goals_scored.std(),
                home_goals_conceded.min(),
                home_goals_conceded.max(),
                home_goals_conceded.mean(),
                home_goals_conceded.median(),
                home_goals_conceded.std(),
                away_short_name,
                int(away[1][:-2]),
                running_stats['away'][away_short_name]['matches_played'],
                running_stats['away'][away_short_name]['goals_scored_total'],
                running_stats['away'][away_short_name]['goals_conceded_total'],
                away_goals_scored.min(),
                away_goals_scored.max(),
                away_goals_scored.mean(),
                away_goals_scored.median(),
                away_goals_scored.std(),
                away_goals_conceded.min(),
                away_goals_conceded.max(),
                away_goals_conceded.mean(),
                away_goals_conceded.median(),
                away_goals_conceded.std(),
            ]

            list_matches.append(this_match)

            running_stats['home'][home_short_name]['goals_scored'].append(home_goals)
            running_stats['home'][home_short_name]['goals_conceded'].append(away_goals)
            running_stats['home'][home_short_name]['goals_scored_total'] = running_stats['home'][home_short_name][
                                                                               'goals_scored_total'] + home_goals
            running_stats['home'][home_short_name]['goals_conceded_total'] = running_stats['home'][home_short_name][
                                                                                 'goals_conceded_total'] + away_goals
            running_stats['home'][home_short_name]['matches_played'] = running_stats['home'][home_short_name][
                                                                           'matches_played'] + 1

            running_stats['away'][away_short_name]['goals_scored'].append(away_goals)
            running_stats['away'][away_short_name]['goals_conceded'].append(home_goals)
            running_stats['away'][away_short_name]['goals_scored_total'] = running_stats['away'][away_short_name][
                                                                               'goals_scored_total'] + away_goals
            running_stats['away'][away_short_name]['goals_conceded_total'] = running_stats['away'][away_short_name][
                                                                                 'goals_conceded_total'] + home_goals
            running_stats['away'][away_short_name]['matches_played'] = running_stats['away'][away_short_name][
                                                                           'matches_played'] + 1

    columns = [
        'gameweek', 'match_date',
        'home_team', 'h_rank', 'h_matches_played', 'h_goals_s_sum', 'h_goals_c_sum',
        'h_goals_s_min', 'h_goals_s_max', 'h_goals_s_mean', 'h_goals_s_median', 'h_goals_s_std',
        'h_goals_c_min', 'h_goals_c_max', 'h_goals_c_mean', 'h_goals_c_median', 'h_goals_c_std',
        'away_team', 'a_rank', 'a_matches_played', 'a_goals_s', 'a_goals_c',
        'a_goals_s_min', 'a_goals_s_max', 'a_goals_s_mean', 'a_goals_s_median', 'a_goals_s_std',
        'a_goals_c_min', 'a_goals_c_max', 'a_goals_c_mean', 'a_goals_c_median', 'a_goals_c_std'
    ]
    df_matches = pd.DataFrame(list_matches, columns=columns)
    df_matches['match_date'] = df_matches['match_date'].apply(pd.to_datetime)
    # print(df_matches[['h_rank', 'h_matches_played', 'h_goals_s', 'h_goals_c']].head(n=10))
    return df_matches


def _created_combined_db(season, league_id, df_matches):
    filename = 'kaggle_db/features_{season}_{league}_{version}.csv'
    df_otherdb_data = pd.read_csv(filename.format(season=season.replace('/', '_'), league=league_id, version='v2_all'))

    # print('scraped data: ', df_matches.shape)
    # print('kaggle data: ', df_otherdb_data.shape)

    df_combined_data = pd.concat([df_otherdb_data, df_matches], axis=1)

    # print('merged data: ', df_combined_data.shape)
    # print(df_combined_data[['gameweek', 'match_date', 'home_rank', 'away_rank', 'hp_weight_mean']].head(n=10))
    # df_combined_data_na = df_combined_data[df_combined_data.isnull().any(axis=1)]
    # print(df_combined_data_na[['gameweek', 'match_date', 'home_team', 'away_team']].head(n=10))

    df_combined_data_without_nulls = df_combined_data.dropna()
    return df_combined_data_without_nulls


def main():
    source_matches = [
        {'season': '2008/2009', 'league_id': 1729},
        {'season': '2009/2010', 'league_id': 1729},
        {'season': '2010/2011', 'league_id': 1729},
        {'season': '2011/2012', 'league_id': 1729},
        {'season': '2012/2013', 'league_id': 1729},
        {'season': '2013/2014', 'league_id': 1729},
        {'season': '2014/2015', 'league_id': 1729},
        {'season': '2015/2016', 'league_id': 1729},
    ]

    for source in source_matches:
        print('Starting extraction... [season={season}, league: {league}]'.format(
            season=source['season'], league=source['league_id']))
        df_matches = _extract_matchday_team_ranks(source['season'], source['league_id'])
        df_combined_data = _created_combined_db(source['season'], source['league_id'], df_matches)
        filename = 'combined_db/features_{season}_{league}_{version}.csv'.format(
            season=source['season'].replace('/', '_'), league=source['league_id'], version='v2_all')
        df_combined_data.to_csv(filename)
        print('Extraction complete... [(matches, feature_count): {shape} saved to file: {filename}]'.format(
            records=df_combined_data.size, shape=df_combined_data.shape, filename=filename))


if __name__ == '__main__':
    main()
