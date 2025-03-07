-- a SQL script that creates a table users

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    country VARCHAR(2) NOT NULL DEFAULT 'US' CHECK (country IN ('US', 'CO', 'TN')),
    PRIMARY KEY (id)
);