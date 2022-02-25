import requests
from bs4 import BeautifulSoup
import pandas as pd

"""Get reviews and store them in a file, the only problem is, it might not get 
all the reviews present on steam, but maybe 60-75% of them. Also, if there 
is an error, just run it again, that should work ig """

"""You only need to set the 4 parameters, game_name, number, sentiment, file_name 
and run the program. Also put both the preprocessing and this program in the same folder"""


"""This is just the name of the game"""
#game_name = "Cyberpunk 2077"
game_name = "Witcher 3"

"""This is the number of reviews you want to get"""
number = 200000

"""This can either be "positive" or "negative", i kept them seperate so that 
we can get a proper amount of both """
sentiment = "positive"

"""Name of the file to store the data, keep positive and negative seperately, 
we will join Them in the preprocessing step anyway"""
file_name = "witcher_pos"


def get_app_id(game_name):
    '''Input the game's name you want to get it's id to be used to get reviews'''
    response = requests.get(url=f'https://store.steampowered.com/search/?term={game_name}&category1=998', headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    app_id = soup.find(class_='search_result_row')['data-ds-appid']
    return app_id

def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

def get_n_reviews(appid, n=100, review_type='all'):
    '''
    Here, review_type can be 'all', 'positive', or 'negative'. So use this 
    filter to get the ideal number of positive and negative reviews
    '''
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : review_type,
            'purchase_type' : 'all'
            }
    
    
    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        #if len(response['reviews']) < 100: break

    return reviews

app_id = get_app_id(game_name)
#print(app_id)
reviews = []
old = 0
check = 1
print(app_id)
while (True):
    reviews = get_n_reviews(app_id, number, sentiment)
    print(len(reviews))
    new = len(reviews)
    if (new==old) and (check==0):
        check = 1
    elif (new==old) and (check==1):
        break
    old = new

#print(reviews[800])
df = pd.DataFrame(reviews)
file_name = file_name + ".csv"
df.to_csv(file_name)
