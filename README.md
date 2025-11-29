# hive

To get it up and running do the following:-

1) using the env example setup the variables in a .env file in the main directory

2) Set up the neon db server from the init_db thing manually

To get the containers up and running:-

3) ```docker compose up --build``` (first time could take 10-20 mins to build)

4) The frontend will now be accessible on http://0.0.0.0:8501/

5) The frontend is fairly intuitive to operate. No backend knowledge is needed to operate it.

6) For some reason it appears to only work on chromium based browsers and not on Firefox.

7) Testing includes individual tests for api_service and audio_service for debugging purposes.

8) All the LLM evaluations are present under the folder ```evals```.

-----------------------------------------
Project Demo Video is hosted on OneDrive: https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/vijayba_iisc_ac_in/EXokj3-yWPdEqW5MUKy3OZMBADMWQXUT8PtwZ3eTrNxJqQ?e=Li414r

Look at the ```project_overview.txt``` and ```ADSAI_Hive_Project_Report.pdf``` file for more details
