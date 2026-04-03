import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [accounts, setAccounts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [findings, setFindings] = useState(null);
  const [sar, setSar] = useState("");
  const [loadingSAR, setLoadingSAR] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/high-risk-accounts")
      .then((res) => res.json())
      .then((data) => setAccounts(data))
      .catch((err) => console.error("Backend error:", err));
  }, []);

  const selectAccount = async (acc) => {
    setSelected(acc);
    setSar("");
    setFindings(null);

    const res = await fetch(
      `http://localhost:8000/api/account-findings/${acc.account_id}`
    );
    const data = await res.json();
    setFindings(data);
  };

  const generateSAR = async () => {
    if (!selected) return;
    setLoadingSAR(true);
    setSar("");

    const res = await fetch(
      `http://localhost:8000/api/generate-sar/${selected.account_id}`
    );
    const data = await res.json();
    setSar(data.sar);
    setLoadingSAR(false);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🏦 Hybrid AML Intelligence Platform</h1>
        <p>Explainable AI • Graph Analytics • GenAI SAR Automation</p>
      </header>

      <div className="main-layout">
        {/* LEFT PANEL */}
        <div className="left-panel">
          <div className="panel-header">
            ⚠️ High-Risk Accounts
          </div>

          <div className="account-table">
            <div className="table-head">
              <span>Account</span>
              <span>Risk</span>
              <span>Typology</span>
            </div>

            {accounts.map((acc, i) => (
              <div
                key={i}
                className={`table-row ${
                  selected?.account_id === acc.account_id ? "active" : ""
                }`}
                onClick={() => selectAccount(acc)}
              >
                <span className="account-id">{acc.account_id}</span>
                <span className="risk">
                  {(acc.anomaly_score * 100).toFixed(1)}%
                </span>
                <span className="typology">{acc.typology}</span>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="right-panel">
          {!selected ? (
            <div className="placeholder">
              <h2>Select an account to begin AML investigation</h2>
            </div>
          ) : (
            <>
              <div className="case-header">
                🔍 Case Review: {selected.account_id}
              </div>

              {/* Explainability Card */}
              <div className="card">
                <div className="card-title">
                  🧠 Model Explainability (Audit Trail)
                </div>
                <pre className="json-box">
                  {findings
                    ? JSON.stringify(findings, null, 2)
                    : "Loading forensic findings..."}
                </pre>
              </div>

              {/* SAR Action */}
              <div className="sar-section">
                <button className="sar-btn" onClick={generateSAR}>
                  {loadingSAR
                    ? "Generating Regulator-Grade SAR..."
                    : "📄 Generate Full SAR Narrative (GenAI)"}
                </button>
              </div>

              {/* SAR Output */}
              {sar && (
                <div className="card">
                  <div className="card-title">
                    📑 Suspicious Activity Report (AI Generated)
                  </div>
                  <div className="sar-box">{sar}</div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

