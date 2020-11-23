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

            away_short_name = TEAM_SHORT_NAME_LOOKUP.get(away[0].strip(), away[0].strip())
            if away_short_name == away[0].strip() and away_short_name not in ['QPR']:
                print('Short name not found', away_short_name)

            list_matches.append([
                stage,
                date,
                home_short_name,
                int(home[0][1:-1]),
                away_short_name,
                int(away[1][:-2])
            ])
            # print('{stage} | ({hr}) *{home}* [{score}] *{away}* ({ar})'.format(
            #     stage=stage, hr=home[0][1:-1], home=home[1].strip(),
            #     score=match[4].text.strip(), away=away[0].strip(), ar=away[1][:-2]))

    columns = ['gameweek', 'match_date', 'home_team', 'home_rank', 'away_team', 'away_rank']
    df_matches = pd.DataFrame(list_matches, columns=columns)
    df_matches['match_date'] = df_matches['match_date'].apply(pd.to_datetime)
    # print(df_matches[['gameweek', 'match_date', 'home_rank', 'away_rank']].head(n=10))
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
            season=source['season'].replace('/', '_'), league=source['league_id'], version='v1_all')
        df_combined_data.to_csv(filename)
        print('Extraction complete... [(matches, feature_count): {shape} saved to file: {filename}]'.format(
            records=df_combined_data.size, shape=df_combined_data.shape, filename=filename))


if __name__ == '__main__':
    main()
