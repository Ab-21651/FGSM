"use client";

import { useState, useRef, ChangeEvent, FormEvent, useEffect } from "react";
import { useRouter } from "next/navigation";

/* â”€â”€ Types â”€â”€ */
interface AttackResult {
  clean_prediction: number;
  adversarial_prediction: number;
  adversarial_image_base64: string;
  attack_success: boolean;
  epsilon: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function AttackPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.15);
  const [result, setResult] = useState<AttackResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setMounted(true); }, []);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) { setError("Please upload an image first."); return; }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("epsilon", epsilon.toString());

      const response = await fetch(`${API_URL}/attack`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Server error: ${response.status}`);
      }

      const data: AttackResult = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setEpsilon(0.15);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  /* Epsilon bar color helper */
  const epsPercent = (epsilon / 0.5) * 100;
  const epsColor =
    epsilon < 0.1 ? "#2ecc71" : epsilon < 0.25 ? "#237a4a" : "#c94040";

  if (!mounted) return null;

  return (
    <div className="min-h-screen" style={{ background: "#070d0a" }}>
      {/* â”€â”€ Top Nav â”€â”€ */}
      <nav
        className="animate-fade-down sticky top-0 z-50 border-b backdrop-blur-md"
        style={{
          borderColor: "#1a3526",
          background: "rgba(7,13,10,0.85)",
        }}
      >
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-2 group cursor-pointer"
          >
            <svg
              className="w-5 h-5 transition-transform group-hover:-translate-x-1"
              fill="none"
              stroke="#2ecc71"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            <span className="text-sm font-medium" style={{ color: "#2ecc71" }}>
              Back
            </span>
          </button>

          <h1 className="text-lg font-bold tracking-tight" style={{ color: "#2ecc71" }}>
            FGSM Attack Lab
          </h1>

          <span
            className="rounded-full px-3 py-1 text-xs font-medium border"
            style={{
              color: "#2ecc71",
              borderColor: "rgba(46,204,113,0.3)",
              background: "rgba(46,204,113,0.08)",
            }}
          >
            MNIST
          </span>
        </div>
      </nav>

      <div className="mx-auto max-w-6xl px-6 py-8 space-y-8">
        {/* â”€â”€ Upload & Controls Card â”€â”€ */}
        <form onSubmit={handleSubmit}>
          <div
            className="animate-fade-up rounded-2xl border p-6"
            style={{ borderColor: "#1a3526", background: "#111e16" }}
          >
            {/* Section Title */}
            <div className="flex items-center gap-3 mb-6">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(46,204,113,0.15)" }}
              >
                <svg className="w-5 h-5" fill="none" stroke="#2ecc71" strokeWidth={1.5} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold" style={{ color: "#2ecc71" }}>
                  Configure Attack
                </h2>
                <p className="text-xs" style={{ color: "#7ecda0" }}>
                  Upload a digit image and adjust perturbation strength
                </p>
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              {/* â”€â”€ File Upload Zone â”€â”€ */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="relative flex flex-col items-center justify-center h-56 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 group animate-border-pulse"
                style={{
                  borderColor: preview ? "#2ecc71" : "#1a3526",
                  background: preview ? "rgba(46,204,113,0.05)" : "rgba(7,13,10,0.5)",
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  if (f) {
                    setFile(f);
                    setPreview(URL.createObjectURL(f));
                    setResult(null);
                    setError(null);
                  }
                }}
              >
                {preview ? (
                  <div className="relative animate-scale-in">
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-h-44 rounded-lg object-contain"
                    />
                    <div
                      className="absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white cursor-pointer"
                      style={{ background: "#2ecc71" }}
                      onClick={(e) => {
                        e.stopPropagation();
                        resetForm();
                      }}
                    >
                      x
                    </div>
                  </div>
                ) : (
                  <div className="text-center group-hover:scale-105 transition-transform duration-300">
                    <div
                      className="mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 animate-float"
                      style={{ background: "rgba(46,204,113,0.1)" }}
                    >
                      <svg className="w-7 h-7" fill="none" stroke="#2ecc71" strokeWidth={1.5} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5a2.25 2.25 0 002.25-2.25V5.25a2.25 2.25 0 00-2.25-2.25H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium" style={{ color: "#2ecc71" }}>
                      Drop your image here or click to browse
                    </p>
                    <p className="text-xs mt-1" style={{ color: "#7ecda0" }}>
                      PNG or JPEG &middot; 28x28 recommended
                    </p>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>

              {/* â”€â”€ Epsilon + Actions â”€â”€ */}
              <div className="flex flex-col justify-between">
                <div>
                  <label className="block text-sm font-semibold mb-3" style={{ color: "#2ecc71" }}>
                    Epsilon ( &epsilon; ) &mdash; Perturbation Strength
                  </label>

                  {/* Custom slider track */}
                  <div className="relative mb-2">
                    <input
                      type="range"
                      min={0}
                      max={0.5}
                      step={0.01}
                      value={epsilon}
                      onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                      className="w-full h-2 rounded-full appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, ${epsColor} ${epsPercent}%, #1a3526 ${epsPercent}%)`,
                      }}
                    />
                  </div>

                  {/* Epsilon display */}
                  <div className="flex justify-between items-center mb-4">
                    <span className="text-xs" style={{ color: "#7ecda0" }}>0.0</span>
                    <div
                      className="rounded-lg px-4 py-2 font-mono text-xl font-bold transition-colors duration-300"
                      style={{
                        color: epsColor,
                        background: "rgba(46,204,113,0.08)",
                        border: `1px solid ${epsColor}33`,
                      }}
                    >
                      {epsilon.toFixed(2)}
                    </div>
                    <span className="text-xs" style={{ color: "#7ecda0" }}>0.5</span>
                  </div>

                  {/* Epsilon guide */}
                  <div
                    className="rounded-lg p-3 text-xs leading-relaxed"
                    style={{ background: "rgba(46,204,113,0.06)", color: "#7ecda0" }}
                  >
                    <span className="font-semibold" style={{ color: "#2ecc71" }}>Tip:</span>{" "}
                    Low values (0.05â€“0.1) are nearly invisible. Medium (0.15â€“0.25)
                    often fool the model. High (&gt;0.3) creates visible noise.
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 mt-6">
                  <button
                    type="submit"
                    disabled={loading || !file}
                    className="btn-primary flex-1 rounded-xl px-6 py-3.5 text-sm font-semibold text-white disabled:opacity-30 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none cursor-pointer"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                        </svg>
                        Generating...
                      </span>
                    ) : (
                      "Run FGSM Attack"
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={resetForm}
                    className="btn-outline rounded-xl px-5 py-3.5 text-sm font-medium cursor-pointer"
                  >
                    Reset
                  </button>
                </div>
              </div>
            </div>
          </div>
        </form>

        {/* â”€â”€ Error â”€â”€ */}
        {error && (
          <div
            className="animate-scale-in rounded-xl border px-5 py-4 text-sm"
            style={{
              borderColor: "rgba(201,64,64,0.3)",
              background: "rgba(201,64,64,0.08)",
              color: "#e05555",
            }}
          >
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* â”€â”€ Results â”€â”€ */}
        {result && (
          <div className="space-y-6">
            {/* Status Banner */}
            <div
              className={`animate-scale-in rounded-xl border px-6 py-5 text-center ${
                result.attack_success ? "" : ""
              }`}
              style={{
                borderColor: result.attack_success
                  ? "rgba(201,64,64,0.3)"
                  : "rgba(46,204,113,0.3)",
                background: result.attack_success
                  ? "rgba(201,64,64,0.08)"
                  : "rgba(46,204,113,0.08)",
              }}
            >
              <div
                className="text-3xl mb-2 font-bold"
                style={{
                  color: result.attack_success ? "#e05555" : "#2ecc71",
                }}
              >
                {result.attack_success
                  ? "Attack Successful"
                  : "Model is Robust"}
              </div>
              <p
                className="text-sm"
                style={{
                  color: result.attack_success ? "#c07070" : "#7ecda0",
                }}
              >
                {result.attack_success
                  ? `The model was fooled! Prediction changed from ${result.clean_prediction} to ${result.adversarial_prediction}`
                  : `The model correctly predicted ${result.clean_prediction} even after perturbation`}
              </p>
            </div>

            {/* Side-by-side Images */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* Original */}
              <div
                className="animate-slide-left rounded-2xl border p-6 card-hover"
                style={{ borderColor: "#1a3526", background: "#111e16" }}
              >
                <div className="flex items-center gap-2 mb-4">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: "#2ecc71" }}
                  />
                  <h3 className="text-sm font-semibold" style={{ color: "#2ecc71" }}>
                    Original Image
                  </h3>
                </div>
                {preview && (
                  <div className="flex justify-center mb-5">
                    <div
                      className="rounded-xl p-3"
                      style={{ background: "#0a1410" }}
                    >
                      <img
                        src={preview}
                        alt="Original"
                        className="h-48 w-48 rounded-lg object-contain"
                      />
                    </div>
                  </div>
                )}
                <div className="text-center">
                  <span
                    className="text-xs uppercase tracking-widest font-medium"
                    style={{ color: "#7ecda0" }}
                  >
                    Clean Prediction
                  </span>
                  <p
                    className="text-6xl font-bold mt-2"
                    style={{ color: "#2ecc71" }}
                  >
                    {result.clean_prediction}
                  </p>
                </div>
              </div>

              {/* Adversarial */}
              <div
                className="animate-slide-right rounded-2xl border p-6 card-hover"
                style={{
                  borderColor: result.attack_success
                    ? "rgba(201,64,64,0.2)"
                    : "#1a3526",
                  background: result.attack_success
                    ? "rgba(201,64,64,0.04)"
                    : "#111e16",
                }}
              >
                <div className="flex items-center gap-2 mb-4">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{
                      background: result.attack_success ? "#c94040" : "#2ecc71",
                    }}
                  />
                  <h3
                    className="text-sm font-semibold"
                    style={{
                      color: result.attack_success ? "#e05555" : "#2ecc71",
                    }}
                  >
                    Adversarial Image
                  </h3>
                  <span
                    className="ml-auto text-xs rounded-full px-2 py-0.5 font-mono"
                    style={{
                      color: "#7ecda0",
                      background: "rgba(46,204,113,0.1)",
                    }}
                  >
                    &epsilon; = {result.epsilon.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-center mb-5">
                  <div
                    className="rounded-xl p-3"
                    style={{ background: "#0a1410" }}
                  >
                    <img
                      src={`data:image/png;base64,${result.adversarial_image_base64}`}
                      alt="Adversarial"
                      className="h-48 w-48 rounded-lg object-contain"
                    />
                  </div>
                </div>
                <div className="text-center">
                  <span
                    className="text-xs uppercase tracking-widest font-medium"
                    style={{ color: "#7ecda0" }}
                  >
                    Adversarial Prediction
                  </span>
                  <p
                    className="text-6xl font-bold mt-2"
                    style={{
                      color: result.attack_success ? "#e05555" : "#2ecc71",
                    }}
                  >
                    {result.adversarial_prediction}
                  </p>
                </div>
              </div>
            </div>

            {/* Details Table */}
            <div
              className="animate-fade-up rounded-2xl border overflow-hidden"
              style={{ borderColor: "#1a3526", background: "#111e16" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "#1a3526", background: "rgba(46,204,113,0.06)" }}
              >
                <h3 className="text-sm font-semibold" style={{ color: "#2ecc71" }}>
                  Attack Summary
                </h3>
              </div>
              <table className="w-full text-sm">
                <tbody>
                  {[
                    { label: "Epsilon", value: result.epsilon.toFixed(2) },
                    { label: "Clean Prediction", value: result.clean_prediction },
                    { label: "Adversarial Prediction", value: result.adversarial_prediction },
                  ].map((row, i) => (
                    <tr
                      key={i}
                      className="border-b"
                      style={{ borderColor: "rgba(26,53,38,0.5)" }}
                    >
                      <td className="px-5 py-3" style={{ color: "#7ecda0" }}>
                        {row.label}
                      </td>
                      <td
                        className="px-5 py-3 text-right font-mono font-semibold"
                        style={{ color: "#2ecc71" }}
                      >
                        {row.value}
                      </td>
                    </tr>
                  ))}
                  <tr>
                    <td className="px-5 py-3" style={{ color: "#7ecda0" }}>
                      Attack Status
                    </td>
                    <td className="px-5 py-3 text-right">
                      <span
                        className="rounded-full px-3 py-1 text-xs font-semibold"
                        style={{
                          color: result.attack_success ? "#e05555" : "#2ecc71",
                          background: result.attack_success
                            ? "rgba(201,64,64,0.12)"
                            : "rgba(46,204,113,0.12)",
                        }}
                      >
                        {result.attack_success ? "Fooled" : "Robust"}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer
          className="text-center text-xs pt-8 pb-6 border-t"
          style={{ borderColor: "#1a3526", color: "#1a3526" }}
        >
          FGSM Adversarial Attack Demo &middot; Next.js + FastAPI + PyTorch
        </footer>
      </div>
    </div>
  );
}

