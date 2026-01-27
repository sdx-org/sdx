import React from 'react';
import { Row, Col } from 'react-bootstrap';

export default function DemographicsTab({ data }) {
  if (!data) return <p className="text-muted">No data available</p>;

  const { age, gender, weight, height } = data;

  return (
    <div>
      <Row className="g-3">
        <Col md={6}>
          <p>
            <strong>Age:</strong> {age ? `${age} years` : 'Not provided'}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Gender:</strong> {gender || 'Not provided'}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Weight:</strong> {weight ? `${weight} kg` : 'Not provided'}
          </p>
        </Col>
        <Col md={6}>
          <p>
            <strong>Height:</strong> {height ? `${height} cm` : 'Not provided'}
          </p>
        </Col>
      </Row>
    </div>
  );
}
