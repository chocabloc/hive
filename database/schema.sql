-- This is the implementation of the schema discussed in the meet
-- The vector DB part is not included in here yet

CREATE TABLE persons(
    person_id SERIAL PRIMARY KEY,
    person_name TEXT NOT NULL
);

CREATE TABLE locations(
    location_id SERIAL PRIMARY KEY,
    country TEXT NOT NULL,
    city TEXT NOT NULL
);

CREATE TABLE actions(
    action_id SERIAL PRIMARY KEY,
    action_name TEXT NOT NULL,
    action_date DATE NOT NULL,

    -- Many to one links mentioned in the schema

    location_id INT REFERENCES locations(location_id)
);

CREATE TABLE lines(
    line_id SERIAL PRIMARY KEY,
    line_text TEXT NOT NULL,
    spokentime TIMESTAMPZ NOT NULL,

    -- Many to one links mentioned in the schema

    person_id INT REFERENCES persons(person_id),
    action_id INT REFERENCES actions(action_id),
    event_id INT REFERENCES events(event_id),
    location_id INT REFERENCES locations(location_id)
);

CREATE TABLE events(
    event_id SERIAL PRIMARY KEY,
    event_name TEXT NOT NULL,

    -- Many to one links mentioned in the schema

    location_id INT REFERENCES locations(location_id)
);

-- Many to many links mentioned in the schema (Linking Tables)

-- persons to events

CREATE TABLE person_events (
    person_id INT NOT NULL REFERENCES persons(person_id),
    event_id INT NOT NULL REFERENCES events(event_id),
    PRIMARY KEY (person_id, event_id)
);

-- persons to actions

CREATE TABLE person_actions (
    person_id INT NOT NULL REFERENCES persons(person_id),
    action_id INT NOT NULL REFERENCES actions(action_id),
    PRIMARY KEY (person_id, action_id)
);

