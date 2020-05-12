# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_render_template]
import datetime

from flask import Flask, render_template
from google.cloud import datastore

datastore_client = datastore.Client()

def store_time(dt):
    """
    uses the Datastore client libraries to create a new entity in Datastore. 
    Datastore entities are data objects that consist of keys and properties. 
    In this case, the entity's key is its custom kind, 'visit'. The entity 
    also has one property, 'timestamp', containing time of a page request
    """
    entity = datastore.Entity(key=datastore_client.key('visit'))
    entity.update({
        'timestamp': dt
    })

    datastore_client.put(entity)


def fetch_times(limit):
    """
    The fetch_times method uses the key visit to query the database for the 
    'limit' most recent visit entities and then stores those entities in a list  
    in descending order.
    """
    query = datastore_client.query(kind='visit')
    query.order = ['-timestamp']

    times = query.fetch(limit=limit)

    return times


app = Flask(__name__)


@app.route('/')
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    # dummy_times = [datetime.datetime(2018, 1, 1, 10, 0, 0),
    #                datetime.datetime(2018, 1, 2, 10, 30, 0),
    #                datetime.datetime(2018, 1, 3, 11, 0, 0),
    #                ]
    # dummy_times = ['i\'m',
    #                'a',
    #                'bird...',
    #                ]

    # Store the current access time in Datastore.
    store_time(datetime.datetime.now())

    # Fetch the most recent 10 access times from Datastore.
    times = fetch_times(10)

    indicators = range(10)

    times_indicators = []

    for x, y in zip(times, indicators):
        times_indicators.append(f'{y}: {x["timestamp"]}')

    times_indicators = iter(times_indicators)

    return render_template(
        'cbmap1.html', times=times_indicators)

    #return render_template('index.html', times=dummy_times) 


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
