import { useState, useEffect} from 'react';
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
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';


export default function Diagnosis() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { state, dispatch } = useConsultation();

  const {
    handleSubmit,
    formState: { isSubmitting },
  } = useForm();

  const [apiError, setApiError] = useState(null);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
  const [aiSummary, setAiSummary] = useState('');

  // local state for components UI
  const [diagnosisItems, setDiagnosisItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [evaluations, setEvaluations] = useState({});
  const [manualText, setManualText] = useState('');
  const ratingFields = ['accuracy', 'relevance', 'usefulness', 'coherence'];
  const ratingLabels = {
    accuracy: t('ratings.accuracy'),
    relevance: t('ratings.relevance'),
    usefulness: t('ratings.usefulness'),
    coherence: t('ratings.coherence'),
  };

  useEffect(() => {
    const hasExistingData=
      state.formData.diagnosis.suggestions.length>0 ||
      state.formData.diagnosis.selected.length>0 ||
      Object.keys(state.formData.diagnosis.evaluations).length>0;

      if(hasExistingData){
        restoreFromContext();
      }else{
        fetchDiagnosisSuggestions();
      }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[]);


  const restoreFromContext=()=>{
    try{
      setIsLoadingSuggestions(true);
      const aiItems=state.formData.diagnosis.suggestions.map((opt,idx)=>({
        id: opt.id || `ai-${idx}`,
        name: opt.name,
        description: opt.description,
        source: opt.source || 'ai',
        expanded: false,
      }));

      setDiagnosisItems(aiItems);
      setSelectedIds(state.formData.diagnosis.selected);
      setEvaluations(state.formData.diagnosis.evaluations);
      setIsLoadingSuggestions(false);
    }catch(err){
       console.error('Error restoring diagnosis from context:', err);
       fetchDiagnosisSuggestions();
    }
  };

  const fetchDiagnosisSuggestions = async () => {
    try {
      setIsLoadingSuggestions(true);
      setApiError(null);
      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
      }
      const response = await consultationAPI.getDiagnosisSuggestions(state.patientId);
      const options = response.options || [];
      setAiSummary(response.summary || '');

      const aiItems = options.map((opt, idx) => ({
        id: opt.id || `ai-${idx}`,
        name: opt.name,
        description: opt.description,
        source: 'ai',
        expanded: false,
      }));

      setDiagnosisItems(aiItems);

      // Persist suggestions to context
      dispatch(
        consultationActions.updateDiagnosisSuggestions(
          aiItems.map((item)=>({
            id: item.id,
            name:item.name,
            description:item.description,
            source: item.source
          }))
        )
      );
    } catch (err) {
      console.error('Error fetching diagnosis suggestions:', err);
      setApiError(err.message || t('errors.loadDiagnosisFailed'));
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
    setDiagnosisItems((prev) =>
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

  // Requires 4 fields (accuracy, relevance, usefulness, coherence) 1-10
  const isEvaluationComplete = (id) => {
    const ev = evaluations[id];
    if (!ev) return false;
    const fields = [
      ev.accuracy,
      ev.relevance,
      ev.usefulness,
      ev.coherence,
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
    ].filter((v) => typeof v === 'number');

    if (!vals.length) return 0;
    return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
  };


  const handleAddManualDiagnosis = () => {
    const trimmed = manualText.trim();
    if (!trimmed) return;

    const newId = `manual-${Date.now()}`;
    const newItem = {
      id: newId,
      name: trimmed,
      description: '',
      source: 'manual',
      expanded: true, // auto-expand
    };

    setDiagnosisItems((prev) => [...prev, newItem]);
    setSelectedIds((prev) => [...prev, newId]);
    setManualText('');

    dispatch(
      consultationActions.updateDiagnosisSuggestions([
        ...state.formData.diagnosis.suggestions,
        {
          id:newId,
          name: trimmed,
          description: '',
          source: 'manual',
        }
      ])
    );
  };


  const onSubmit = async () => {
    try {
      setApiError(null);

      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
      }

      if (selectedIds.length === 0) {
        setApiError(t('diagnosis.errors.selectAtLeastOne'));
        return;
      }

      const incomplete = selectedIds.filter((id) => !isEvaluationComplete(id));
      if (incomplete.length > 0) {
        const names = diagnosisItems
          .filter((item) => incomplete.includes(item.id))
          .map((item) => item.name);
        setApiError(
          t('diagnosis.errors.completeRatings', { names: names.join(', ') })
        );
        return;
      }

      const selectedNames = diagnosisItems
        .filter((item) => selectedIds.includes(item.id))
        .map((item) => item.name);

      const evaluationsByName = {};
      diagnosisItems
        .filter((item) => selectedIds.includes(item.id))
        .forEach((item) => {
          evaluationsByName[item.name] = evaluations[item.id];
        });

      // Persist to context
      dispatch(
        consultationActions.updateSelectedDiagnosis(selectedIds, evaluations)
      );

      await consultationAPI.saveDiagnosisSelections(state.patientId, {
        selected_diagnoses: selectedNames,
        evaluations: evaluationsByName,
      });

      dispatch(consultationActions.setCurrentStep('exams'));
      navigate('/exams');
    } catch (err) {
      console.error('Error saving diagnosis selections:', err);
      setApiError(err.message || t('errors.saveDiagnosisFailed'));
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
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 7 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 70 })}
            </small>
          </div>
          <ProgressBar now={70} style={{ height: '8px' }} />
        </div>

        {/* Error */}
        {apiError && (
          <Alert
            variant="danger"
            dismissible
            onClose={() => setApiError(null)}
            className="mb-3"
          >
            <Alert.Heading className="h6 mb-2">
              {t('errors.title')}
            </Alert.Heading>
            <p className="mb-0 small">{apiError}</p>
          </Alert>
        )}

        {/* Loading */}
        {isLoadingSuggestions ? (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-5 text-center">
              <Spinner animation="border" variant="primary" className="mb-3" />
              <p className="text-muted mb-0">
                {t('diagnosis.loading')}
              </p>
            </Card.Body>
          </Card>
        ) : (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-4 p-md-5">
              {/* Header */}
              <div className="mb-4">
                <h1 className="h3 fw-bold text-primary mb-1">
                  {t('diagnosis.title')}
                </h1>
                <p className="text-muted mb-0 small">
                  {t('diagnosis.subtitle')}
                </p>
              </div>

              {/* AI summary */}
              {aiSummary && (
                <Card className="bg-light border-0 mb-4">
                  <Card.Body className="p-3">
                    <p className="small mb-1 fw-semibold text-muted">
                      {t('diagnosis.aiSummaryTitle')}
                    </p>
                    <p className="small mb-0 text-muted">{aiSummary}</p>
                  </Card.Body>
                </Card>
              )}

              {/* Add manual diagnosis */}
              <div className="mb-4">
                <p className="small fw-semibold mb-2">
                  {t('diagnosis.addManualTitle')}
                </p>
                <InputGroup>
                  <Form.Control
                    placeholder={t('diagnosis.addManualPlaceholder')}
                    value={manualText}
                    onChange={(e) => setManualText(e.target.value)}
                    size="sm"
                  />
                  <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={handleAddManualDiagnosis}
                  >
                    + {t('common.add')}
                  </Button>
                </InputGroup>
              </div>

              {/* Suggestions list */}
              {diagnosisItems.length === 0 ? (
                <Alert variant="info" className="border-0">
                  <p className="small mb-0">
                    {t('diagnosis.empty')}
                  </p>
                </Alert>
              ) : (
                <div className="mb-4">
                  <p className="small fw-semibold mb-2">
                    {t('diagnosis.reviewHint')}
                  </p>

                  {diagnosisItems.map((item) => {
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
                                      {item.source === 'ai'
                                        ? t('diagnosis.sourceAi')
                                        : t('diagnosis.sourceManual')}
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
                                      {t('common.avg')}: <strong>{average}/10</strong>
                                    </div>
                                  )}
                                  {isSelected && (
                                    <Badge
                                      bg={complete ? 'success' : 'warning'}
                                      className="mt-1"
                                    >
                                      {complete ? t('status.complete') : t('status.pending')}
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
                                {t('diagnosis.rateTitle')}
                              </p>

                              {/* Ratings grid */}
                              <div className="row g-2 mb-3">
                                {ratingFields.map(
                                  (field) => (
                                    <div className="col-12 col-md-6" key={field}>
                                      <Form.Label className="small mb-1 text-muted text-capitalize">
                                        {ratingLabels[field]}
                                      </Form.Label>
                                      <Form.Select
                                        size="sm"
                                        value={evalForItem[field] || ''}
                                        onChange={(e) =>
                                          setRating(item.id, field, e.target.value)
                                        }
                                      >
                                        <option value="">
                                          {t('common.select')}
                                        </option>
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
                                  {t('common.commentsOptional')}
                                </Form.Label>
                                <Form.Control
                                  as="textarea"
                                  rows={2}
                                  size="sm"
                                  placeholder={t('common.rationalePlaceholder')}
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
                      {t('diagnosis.selectedSummary', {
                        count: selectedIds.length,
                      })}
                    </p>
                    <ListGroup variant="flush">
                      {selectedIds.map((id) => {
                        const item = diagnosisItems.find((x) => x.id === id);
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
                                {t('common.scoreValue', { score })}
                              </small>
                            </div>
                            <Badge bg={complete ? 'success' : 'warning'}>
                              {complete ? t('status.complete') : t('status.pending')}
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
                <p className="fw-semibold mb-1">{t('diagnosis.howToRateTitle')}</p>
                <ul className="mb-0 ps-3">
                  <li>
                    <strong>{t('ratings.accuracy')}:</strong> {t('diagnosis.howToRate.accuracy')}
                  </li>
                  <li>
                    <strong>{t('ratings.relevance')}:</strong> {t('diagnosis.howToRate.relevance')}
                  </li>
                  <li>
                    <strong>{t('ratings.usefulness')}:</strong> {t('diagnosis.howToRate.usefulness')}
                  </li>
                  <li>
                    <strong>{t('ratings.coherence')}:</strong> {t('diagnosis.howToRate.coherence')}
                  </li>
                </ul>
              </Alert>

              {/* Actions */}
              <div className="d-flex justify-content-between align-items-center pt-3 border-top">
                <Button
                  variant="outline-secondary"
                  onClick={() => navigate(-1)}
                  size="sm"
                  disabled={isSubmitting}
                >
                  ← {t('common.back')}
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
                      <span>{t('common.processing')}</span>
                    </>
                  ) : (
                    <>
                      <span>{t('actions.nextTo', { step: t('steps.exams') })}</span>
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
            {t('diagnosis.footer')}
          </small>
        </div>
      </Container>
    </div>
  );
}
