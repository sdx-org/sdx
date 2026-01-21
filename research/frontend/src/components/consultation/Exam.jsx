import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import {
  Form,
  Button,
  Container,
  Card,
  ProgressBar,
  Alert,
  Spinner,
  ListGroup,
  Badge,
  Collapse,
  InputGroup,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function Exams() {
  const navigate = useNavigate();
  const { state, dispatch } = useConsultation();

  const {
    handleSubmit,
    formState: { isSubmitting },
  } = useForm();

  const [apiError, setApiError] = useState(null);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
  const [aiSummary, setAiSummary] = useState('');

  // Local state for component UI
  const [examItems, setExamItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [evaluations, setEvaluations] = useState({});
  const [manualText, setManualText] = useState('');


  useEffect(() => {
    const hasExistingData =
      state.formData.exams.suggestions.length > 0 ||
      state.formData.exams.selected.length > 0 ||
      Object.keys(state.formData.exams.evaluations).length > 0;

    if (hasExistingData) {
      restoreFromContext();
    } else {
      fetchExamSuggestions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const restoreFromContext = () => {
    try {
      setIsLoadingSuggestions(true);

      const aiItems = state.formData.exams.suggestions.map((opt, idx) => ({
        id: opt.id || `ai-exam-${idx}`,
        name: opt.name,
        description: opt.description,
        source: opt.source || 'ai',  // Preserve original source
        expanded: false,
      }));

      setExamItems(aiItems);
      setSelectedIds(state.formData.exams.selected);
      setEvaluations(state.formData.exams.evaluations);

      setIsLoadingSuggestions(false);
    } catch (err) {
      console.error('Error restoring exams from context:', err);
      fetchExamSuggestions();
    }
  };

  // Fetch fresh AI suggestions
  const fetchExamSuggestions = async () => {
    try {
      setIsLoadingSuggestions(true);
      setApiError(null);

      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      const response = await consultationAPI.getExamSuggestions(state.patientId);
      const options = response.options || [];
      setAiSummary(response.summary || '');

      const aiItems = options.map((opt, idx) => ({
        id: opt.id || `ai-exam-${idx}`,
        name: opt.name,
        description: opt.description,
        source: 'ai',
        expanded: false,
      }));

      setExamItems(aiItems);

      dispatch(
        consultationActions.updateExamSuggestions(
          aiItems.map((item) => ({
            id: item.id,
            name: item.name,
            description: item.description,
            source: item.source
          }))
        )
      );
    } catch (err) {
      console.error('Error fetching exam suggestions:', err);
      setApiError(err.message || 'Failed to load exam suggestions. Please try again.');
    } finally {
      setIsLoadingSuggestions(false);
    }
  };

  const toggleSelected = (id) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const toggleExpanded = (id) => {
    setExamItems((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, expanded: !item.expanded } : item
      )
    );
  };

  const setRating = (id, field, value) => {
    setEvaluations((prev) => ({
      ...prev,
      [id]: {
        ...(prev[id] || {}),
        [field]: Number(value),
      },
    }));
  };

  const setComment = (id, comment) => {
    setEvaluations((prev) => ({
      ...prev,
      [id]: {
        ...(prev[id] || {}),
        comments: comment,
      },
    }));
  };

  // Requires all 5 fields to be set (1-10)
  const isEvaluationComplete = (id) => {
    const ev = evaluations[id];
    if (!ev) return false;
    const fields = [
      ev.accuracy,
      ev.relevance,
      ev.usefulness,
      ev.coherence,
      ev.safety,
    ];
    return fields.every((v) => typeof v === 'number' && v >= 1 && v <= 10);
  };

  const getAverageScore = (id) => {
    const ev = evaluations[id];
    if (!ev) return 0;
    const vals = [
      ev.accuracy,
      ev.relevance,
      ev.usefulness,
      ev.coherence,
      ev.safety,
    ].filter((v) => typeof v === 'number');

    if (!vals.length) return 0;
    return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
  };

  const handleAddManualExam = () => {
    const trimmed = manualText.trim();
    if (!trimmed) return;

    const newId = `manual-exam-${Date.now()}`;
    const newItem = {
      id: newId,
      name: trimmed,
      description: '',
      source: 'manual',
      expanded: true,
    };

    setExamItems((prev) => [...prev, newItem]);
    setSelectedIds((prev) => [...prev, newId]);
    setManualText('');

    dispatch(
      consultationActions.updateExamSuggestions([
        ...state.formData.exams.suggestions,
        { id: newId, name: trimmed, description: '', source: 'manual' }
      ])
    );
  };

  const onSubmit = async () => {
    try {
      setApiError(null);

      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      if (selectedIds.length === 0) {
        setApiError('Please select at least one exam/test.');
        return;
      }

      const incomplete = selectedIds.filter((id) => !isEvaluationComplete(id));
      if (incomplete.length > 0) {
        const names = examItems
          .filter((item) => incomplete.includes(item.id))
          .map((item) => item.name);
        setApiError(`Please complete ratings for: ${names.join(', ')}`);
        return;
      }

      const selectedNames = examItems
        .filter((item) => selectedIds.includes(item.id))
        .map((item) => item.name);

      const evaluationsByName = {};
      examItems
        .filter((item) => selectedIds.includes(item.id))
        .forEach((item) => {
          evaluationsByName[item.name] = evaluations[item.id];
        });

      dispatch(
        consultationActions.updateSelectedExams(selectedIds, evaluations)
      );

      // API Call
      await consultationAPI.saveExamSelections(state.patientId, {
        selected_exams: selectedNames,
        evaluations: evaluationsByName,
      });

      dispatch(consultationActions.completeConsultation());
      dispatch(consultationActions.setCurrentStep('confirmation'));
      navigate('/confirmation');
    } catch (err) {
      console.error('Error saving exam selections:', err);
      setApiError(err.message || 'Failed to save selections. Please try again.');
      window.scrollTo(0, 0);
    }
  };

  const hasSelected = selectedIds.length > 0;

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container style={{ maxWidth: '900px' }}>
        {/* Progress */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 8 of 10</small>
            <small className="text-muted">80% complete</small>
          </div>
          <ProgressBar now={80} style={{ height: '8px' }} />
        </div>

        {/* Error */}
        {apiError && (
          <Alert
            variant="danger"
            dismissible
            onClose={() => setApiError(null)}
            className="mb-3"
          >
            <Alert.Heading className="h6 mb-2">Error</Alert.Heading>
            <p className="mb-0 small">{apiError}</p>
          </Alert>
        )}

        {/* Loading */}
        {isLoadingSuggestions ? (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-5 text-center">
              <Spinner animation="border" variant="primary" className="mb-3" />
              <p className="text-muted mb-0">
                Generating exam and test suggestions…
              </p>
            </Card.Body>
          </Card>
        ) : (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-4 p-md-5">
              {/* Header */}
              <div className="mb-4">
                <h1 className="h3 fw-bold text-primary mb-1">
                  AI Exam / Test Suggestions
                </h1>
                <p className="text-muted mb-0 small">
                  Select recommended exams, rate each one, and add any additional tests.
                </p>
              </div>

              {/* AI summary */}
              {aiSummary && (
                <Card className="bg-light border-0 mb-4">
                  <Card.Body className="p-3">
                    <p className="small mb-1 fw-semibold text-muted">
                      AI exam summary
                    </p>
                    <p className="small mb-0 text-muted">{aiSummary}</p>
                  </Card.Body>
                </Card>
              )}

              {/* Add manual exam */}
              <div className="mb-4">
                <p className="small fw-semibold mb-2">Add manual exam / test</p>
                <InputGroup>
                  <Form.Control
                    placeholder="Type a custom exam/test and click Add"
                    value={manualText}
                    onChange={(e) => setManualText(e.target.value)}
                    size="sm"
                  />
                  <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={handleAddManualExam}
                  >
                    + Add
                  </Button>
                </InputGroup>
              </div>

              {/* Exam list */}
              {examItems.length === 0 ? (
                <Alert variant="info" className="border-0">
                  <p className="small mb-0">
                    No exam suggestions available. Please ensure previous steps are completed.
                  </p>
                </Alert>
              ) : (
                <div className="mb-4">
                  <p className="small fw-semibold mb-2">
                    Review each exam suggestion. Click a row to expand and rate.
                  </p>

                  {examItems.map((item) => {
                    const isSelected = selectedIds.includes(item.id);
                    const evalForItem = evaluations[item.id] || {};
                    const average = getAverageScore(item.id);
                    const complete = isEvaluationComplete(item.id);

                    return (
                      <Card
                        key={item.id}
                        className={`mb-2 border ${
                          isSelected ? 'border-primary' : 'border-light'
                        }`}
                      >
                        <Card.Body className="py-2 px-3">
                          <div className="d-flex align-items-start">
                            {/* Checkbox */}
                            <Form.Check
                              type="checkbox"
                              className="mt-1 me-3"
                              checked={isSelected}
                              onChange={() => toggleSelected(item.id)}
                            />

                            {/* Main content */}
                            <div
                              className="flex-grow-1"
                              style={{ cursor: 'pointer' }}
                              onClick={() => toggleExpanded(item.id)}
                            >
                              <div className="d-flex justify-content-between align-items-start">
                                <div>
                                  <div className="d-flex align-items-center gap-2">
                                    <span className="fw-semibold small">
                                      {item.name}
                                    </span>
                                    <Badge
                                      bg={
                                        item.source === 'ai'
                                          ? 'info'
                                          : 'secondary'
                                      }
                                      pill
                                    >
                                      {item.source === 'ai' ? 'AI' : 'Manual'}
                                    </Badge>
                                  </div>
                                  {item.description && (
                                    <p className="text-muted small mb-0 mt-1">
                                      {item.description}
                                    </p>
                                  )}
                                </div>

                                <div className="text-end">
                                  {average > 0 && (
                                    <div className="small text-muted">
                                      Avg: <strong>{average}/10</strong>
                                    </div>
                                  )}
                                  {isSelected && (
                                    <Badge
                                      bg={complete ? 'success' : 'warning'}
                                      className="mt-1"
                                    >
                                      {complete ? 'Complete' : 'Pending'}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Expandable review area */}
                          <Collapse in={item.expanded}>
                            <div className="mt-3 border-top pt-3">
                              <p className="small fw-semibold mb-2">
                                Rate this exam / test (1–10)
                              </p>

                              {/* Ratings grid */}
                              <div className="row g-2 mb-3">
                                {['accuracy', 'relevance', 'usefulness', 'coherence', 'safety'].map(
                                  (field) => (
                                    <div className="col-12 col-md-6" key={field}>
                                      <Form.Label className="small mb-1 text-muted text-capitalize">
                                        {field}
                                      </Form.Label>
                                      <Form.Select
                                        size="sm"
                                        value={evalForItem[field] || ''}
                                        onChange={(e) =>
                                          setRating(item.id, field, e.target.value)
                                        }
                                      >
                                        <option value="">Select</option>
                                        {Array.from({ length: 10 }).map((_, i) => (
                                          <option key={i + 1} value={i + 1}>
                                            {i + 1}
                                          </option>
                                        ))}
                                      </Form.Select>
                                    </div>
                                  )
                                )}
                              </div>

                              {/* Comments */}
                              <Form.Group className="mb-1">
                                <Form.Label className="small mb-1 text-muted">
                                  Comments (optional)
                                </Form.Label>
                                <Form.Control
                                  as="textarea"
                                  rows={2}
                                  size="sm"
                                  placeholder="Rationale..."
                                  value={evalForItem.comments || ''}
                                  onChange={(e) =>
                                    setComment(item.id, e.target.value)
                                  }
                                />
                              </Form.Group>
                            </div>
                          </Collapse>
                        </Card.Body>
                      </Card>
                    );
                  })}
                </div>
              )}

              {/* Selected summary */}
              {hasSelected && (
                <Card className="bg-light border-0 mb-4">
                  <Card.Body className="p-3">
                    <p className="small fw-semibold mb-2">
                      Selected exams / tests ({selectedIds.length})
                    </p>
                    <ListGroup variant="flush">
                      {selectedIds.map((id) => {
                        const item = examItems.find((x) => x.id === id);
                        if (!item) return null;
                        const score = getAverageScore(id);
                        const complete = isEvaluationComplete(id);
                        return (
                          <ListGroup.Item
                            key={id}
                            className="d-flex justify-content-between align-items-center py-2"
                          >
                            <div>
                              <p className="mb-1 small fw-semibold">{item.name}</p>
                              <small className="text-muted">
                                Score: {score}/10
                              </small>
                            </div>
                            <Badge bg={complete ? 'success' : 'warning'}>
                              {complete ? 'Complete' : 'Pending'}
                            </Badge>
                          </ListGroup.Item>
                        );
                      })}
                    </ListGroup>
                  </Card.Body>
                </Card>
              )}

              {/* Info */}
              <Alert
                variant="info"
                className="border-0 bg-info bg-opacity-10 mb-4 small"
              >
                <p className="fw-semibold mb-1">How to rate exams:</p>
                <ul className="mb-0 ps-3">
                  <li>
                    <strong>Accuracy:</strong> Appropriateness for diagnosis.
                  </li>
                  <li>
                    <strong>Relevance:</strong> Relevance to clinical question.
                  </li>
                  <li>
                    <strong>Usefulness:</strong> Impact on management.
                  </li>
                  <li>
                    <strong>Coherence:</strong> Fit with patient context.
                  </li>
                  <li>
                    <strong>Safety:</strong> Risk profile of the test.
                  </li>
                </ul>
              </Alert>

              {/* Actions */}
              <div className="d-flex justify-content-between align-items-center pt-3 border-top">
                <Button
                  variant="outline-secondary"
                  onClick={() => navigate('/diagnosis')}
                  size="sm"
                  disabled={isSubmitting}
                >
                  ← Back to Diagnosis
                </Button>

                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  disabled={
                    isSubmitting ||
                    !hasSelected ||
                    selectedIds.some((id) => !isEvaluationComplete(id))
                  }
                  onClick={handleSubmit(onSubmit)}
                  className="d-flex align-items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner animation="border" size="sm" />
                      <span>Processing…</span>
                    </>
                  ) : (
                    <>
                      <span>Next: Confirmation</span>
                      <span>→</span>
                    </>
                  )}
                </Button>
              </div>
            </Card.Body>
          </Card>
        )}

        <div className="text-center mt-3">
          <small className="text-muted">
            Your exam selections help optimize AI recommendations.
          </small>
        </div>
      </Container>
    </div>
  );
}
