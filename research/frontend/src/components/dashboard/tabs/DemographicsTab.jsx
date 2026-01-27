import React from 'react';
import { Row, Col } from 'react-bootstrap';

// Helper to format numeric values with units, handling 0 correctly
function formatNumeric(value, unit) {
  return value === null || value === undefined
    ? 'Not provided'
    : `${value} ${unit}`;
}

export default function DemographicsTab({ data }) {
  if (!data) return <p className="text-muted">No data available</p>;

  const { age, gender, weight, height } = data;

  return (
    <div>
      <Row className="g-3">
        <Col md={6}>
          <p>
            <strong>Age:</strong> {formatNumeric(age, 'years')}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Gender:</strong> {gender || 'Not provided'}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Weight:</strong> {formatNumeric(weight, 'kg')}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Height:</strong> {formatNumeric(height, 'cm')}
          </p>
        </Col>
      </Row>
    </div>
  );
}
