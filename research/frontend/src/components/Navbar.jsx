import React, { useEffect } from "react";
import { Navbar, Nav, Container, Dropdown } from "react-bootstrap";
import { Link, NavLink, useInRouterContext } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useConsultation } from "../context/ConsultationContext";

export default function AppNavbar() {
  const { i18n, t } = useTranslation();
  const inRouter = useInRouterContext();
  const { state } = useConsultation();

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    localStorage.setItem("i18nextLng", lng);
  };

  useEffect(() => {
    const saved = localStorage.getItem("i18nextLng");
    if (saved) i18n.changeLanguage(saved);
  }, []);

  const hasResume =
    Boolean(state?.patientId) &&
    Boolean(state?.currentStep) &&
    state.currentStep !== "language-selection" &&
    state.currentStep !== "confirmation" &&
    !state?.isComplete;

  return (
    <Navbar bg="light" expand="lg" className="shadow-sm py-2">
      <Container>
        <Navbar.Brand
          as={inRouter ? Link : "a"}
          to="/"
          href="/"
          className="h5 fw-bold text-primary fs-2"
        >
          HiperHealth
        </Navbar.Brand>

        <Navbar.Toggle aria-controls="navbar-menu" />

        <Navbar.Collapse id="navbar-menu">
          <Nav className="ms-auto align-items-center">

            {/* Dashboard Link */}
            <Nav.Link
              as={inRouter ? NavLink : "a"}
              to="/"
              href="/"
              className="fw-semibold text-dark px-3 small"
            >
              {t("navbar.dashboard", "Patient Dashboard")}
            </Nav.Link>

            {/* Resume Consultation */}
            {hasResume && (
              <Nav.Link
                as={inRouter ? NavLink : "a"}
                to={`/${state.currentStep}`}
                href={`/${state.currentStep}`}
                className="fw-semibold text-dark px-3"
              >
                {t("navbar.resumeDiagnosis", "Resume Diagnosis")}
              </Nav.Link>
            )}

            {/* Language Selector */}
            <Dropdown align="end" className="px-2">
              <Dropdown.Toggle variant="outline-primary" size="sm" className="fw-semibold">
                {i18n.language?.toUpperCase()}
              </Dropdown.Toggle>

              <Dropdown.Menu>
                <Dropdown.Item onClick={() => changeLanguage("en")}>English</Dropdown.Item>
                <Dropdown.Item onClick={() => changeLanguage("es")}>Spanish</Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>

          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}
