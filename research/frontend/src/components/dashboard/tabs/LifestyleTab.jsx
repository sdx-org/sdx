import React from 'react';
import { Row, Col } from 'react-bootstrap';

import PropTypes from 'prop-types';

export default function LifestyleTab({ data }) {
    if (!data) return <p className="text-muted">No data available</p>;

    const { diet, sleep_hours, physical_activity, mental_exercises } = data;

    return (
        <div>
            <Row className="g-3">
                <Col md={6}>
                    <p>
                        <strong>Diet:</strong> {diet ?? 'Not provided'}
                    </p>
                </Col>
                <Col md={6}>
                    <p>
                        <strong>Sleep Hours:</strong>{' '}
                        {sleep_hours != null
                            ? `${sleep_hours} hours/day`
                            : 'Not provided'}
                    </p>
                </Col>
                <Col md={6}>
                    <p>
                        <strong>Physical Activity:</strong>{' '}
                        {physical_activity ?? 'Not provided'}
                    </p>
                </Col>
                <Col md={6}>
                    <p>
                        <strong>Mental Exercises:</strong>{' '}
                        {mental_exercises ?? 'Not provided'}
                    </p>
                </Col>
            </Row>
        </div>
    );
}

/** LifestyleTab PropTypes */
LifestyleTab.propTypes = {
    data: PropTypes.shape({
        diet: PropTypes.string,
        sleep_hours: PropTypes.number,
        physical_activity: PropTypes.string,
        mental_exercises: PropTypes.string,
    }),
};
