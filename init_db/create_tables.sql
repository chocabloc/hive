CREATE TABLE IF NOT EXISTS entities(
    entity_id SERIAL PRIMARY KEY,
    entity_name TEXT NOT NULL 
);
CREATE TABLE IF NOT EXISTS locations( 
    location_id SERIAL PRIMARY KEY, 
    country TEXT NOT NULL, 
    city TEXT NOT NULL 
); 
    
CREATE TABLE IF NOT EXISTS dates ( 
    date_id SERIAL PRIMARY KEY, 
    event_date TEXT NOT NULL 
); 
    
CREATE TABLE IF NOT EXISTS actions( 
    action_id SERIAL PRIMARY KEY, 
    action_name TEXT NOT NULL, 
    date_id INT REFERENCES dates(date_id),
    location_id INT REFERENCES locations(location_id) 
); 
    
CREATE TABLE IF NOT EXISTS events( 
    event_id SERIAL PRIMARY KEY, 
    event_name TEXT NOT NULL, 
    date_id INT REFERENCES dates(date_id),
    location_id INT REFERENCES locations(location_id) 
); 

CREATE TABLE IF NOT EXISTS lines(
    line_id SERIAL PRIMARY KEY, 
    line_text TEXT NOT NULL, 
    spokentime TIMESTAMPTZ NOT NULL, 
    entity_id INT REFERENCES entities(entity_id), 
    action_id INT REFERENCES actions(action_id), 
    event_id INT REFERENCES events(event_id), 
    location_id INT REFERENCES locations(location_id) 
); 
-- Many to many links mentioned in the schema (Linking Tables) 
-- persons to events 

CREATE TABLE IF NOT EXISTS entity_events (
    entity_id INT NOT NULL REFERENCES entities(entity_id),
    event_id INT NOT NULL REFERENCES events(event_id),
    PRIMARY KEY (entity_id, event_id)
);
-- persons to actions 

CREATE TABLE IF NOT EXISTS entity_actions (
    entity_id INT NOT NULL REFERENCES entities(entity_id),
    action_id INT NOT NULL REFERENCES actions(action_id),
    PRIMARY KEY (entity_id, action_id)
);


CREATE EXTENSION IF NOT EXISTS vector; 
-- Summaries table 
CREATE TABLE IF NOT EXISTS summaries ( 
    id SERIAL PRIMARY KEY, 
    summary_text TEXT NOT NULL, 
    
    embedding VECTOR(768) NOT NULL, -- 768-dim embedding vector
    
    created_at TIMESTAMPTZ DEFAULT NOW() 
    
); 
-- Linking table to connect summaries to lines 
CREATE TABLE IF NOT EXISTS summary_lines ( 
    summary_id INT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE, 
    line_id INT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE, 
    PRIMARY KEY (summary_id, line_id) 
    
);