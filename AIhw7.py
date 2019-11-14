#Lucas Manker
#HW7
#11/13/19
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

def pull_data():
    page = requests.get("https://w1.weather.gov/data/obhistory/KLAR.html")
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.findAll('table')[3]
    data = []
    for row in table.findAll('tr'):
        temp = []
        for entry in row.findAll('td'):
            temp.append(entry.text)
        if(temp): data.append(temp)
    return(data)

def get_matrix(data, attrs): 
    i = len(data) - 1
    sum = 0
    for x in range(len(data['weather'])):
        data['weather'][x] = data['weather'][x].lstrip()
    for x in range(len(attrs)):
        attrs[x] = attrs[x].lstrip()
    attrs = np.sort(attrs)
    counts = [0]*len(attrs)
    trans_matrix = a = [[0]*len(attrs) for _ in range(len(attrs))]
    for x in range(len(attrs)):
        counts[x] = (data[data.weather==attrs[x]].shape[0])
    result = (np.where(attrs=='Fair'))
    while(i > 1):
        first = np.where(attrs==data['weather'][i])[0][0]
        second = np.where(attrs==data['weather'][i-1])[0][0]
        trans_matrix[first][second] += 1
        i-=1
    for x in range(len(trans_matrix)):
        trans_matrix[x][:] = [(xs /70)/(counts[x]/70) for xs in trans_matrix[x]]
    title_str = "$X_{t}$ \\textbackslash\\ $X_{t+1}$  "
    for x in attrs:
        title_str = title_str + "&"+x
    print(title_str + "\\\\\\hline")
    temp_string = " "
    for x in range(len(trans_matrix)):
        temp_string += attrs[x]
        for y in trans_matrix[x]:
            temp_string = temp_string + "&" +  str(y)
        print(temp_string, "\\\\\\hline")
        temp_string = ""
    for x in counts:
        sum += x
    return trans_matrix
def get_marg(prob, matrix):
    print(prob.dot(matrix))

def matrix_mult(x,y,matrix):
    sum = 0
    xarr = matrix[x]
    yarr = matrix[y]
    for x in range(len(matrix)):
        sum += xarr[x] * yarr[x]
    return sum
def matrix_loop(x, matrix):
    probs = []
    xarr = matrix[x]
    for y in range(len(matrix)):
        yarr = matrix[y]
        sum = 0
        for x in range(len(matrix)):
            sum += xarr[x] * yarr[x]
        probs.append(sum)
    return probs
def hidden_mark(data):
    weatherS = data.weather.unique()
    dewS = data.dew.unique()
    weatherC = [0]*len(weatherS)
    dewC = [0]*len(dewS)
    for x in range(len(weatherC)):
        weatherC[x] = (data[data.weather==weatherS[x]].shape[0])
    for x in range(len(dewC)):
        dewC[x] = (data[data.dew==dewS[x]].shape[0])
    probs =  [[0]*len(dewS) for _ in range(len(weatherS))]
    for x in range(len(data)):
        x1=(np.where(weatherS==data['weather'][x])[0][0])
        y1=(np.where(dewS==data['dew'][x])[0][0])
        probs[x1][y1] += 1
    sum = 0
    for x in probs:
        for y in x:
            sum += y
    print(sum)


def main():
    orig = pd.DataFrame.from_records(pull_data())
    frame = pd.DataFrame.from_records(pull_data())
    frame.columns=['date','time','wind','vis','weather','sky','air','dew','max','min'
            ,'hum','wind chill','heat','alt','sea level','1hr','3hr','6hr']
    frame.drop(frame.tail(2).index,inplace=True)
    matrix = get_matrix(frame, frame.weather.unique())
    print(matrix_mult(2,2,matrix))
    prob = np.asarray(matrix_loop(2,matrix))
    get_marg(prob, matrix)
    hidden_mark(frame)

if __name__== "__main__":
  main()
