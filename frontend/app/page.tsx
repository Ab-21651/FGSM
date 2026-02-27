"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

/* ── Floating Particle ── */
function Particle({ delay, left, size }: { delay: number; left: string; size: number }) {
  return (
    <div
      className="particle"
      style={{
        left,
        bottom: "-10px",
        width: `${size}px`,
        height: `${size}px`,
        animationDelay: `${delay}s`,
        animationDuration: `${6 + Math.random() * 6}s`,
        opacity: 0.4 + Math.random() * 0.4,
      }}
    />
  );
}

export default function LandingPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="relative min-h-screen overflow-hidden" style={{ background: "#070d0a" }}>
      {/* ── Background Effects ── */}

      {/* Radial gradient glow */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full animate-pulse-glow"
        style={{
          background: "radial-gradient(circle, rgba(46,204,113,0.12) 0%, transparent 70%)",
        }}
      />

      {/* Spinning ring */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full border animate-spin-slow"
        style={{ borderColor: "rgba(46,204,113,0.08)" }}
      />
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[650px] rounded-full border animate-spin-slow"
        style={{
          borderColor: "rgba(46,204,113,0.05)",
          animationDirection: "reverse",
          animationDuration: "30s",
        }}
      />

      {/* Floating particles */}
      {Array.from({ length: 20 }).map((_, i) => (
        <Particle
          key={i}
          delay={i * 0.8}
          left={`${5 + (i * 4.5) % 90}%`}
          size={2 + (i % 4)}
        />
      ))}

      {/* ── Content ── */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 text-center">
        {/* Badge */}
        <div className="animate-fade-down mb-8">
          <span
            className="inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium tracking-wider uppercase border"
            style={{
              color: "#2ecc71",
              borderColor: "rgba(46,204,113,0.3)",
              background: "rgba(46,204,113,0.08)",
            }}
          >
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "#2ecc71" }} />
            Neural Network Security
          </span>
        </div>

        {/* Title */}
        <h1 className="animate-fade-up text-5xl md:text-7xl font-bold tracking-tight leading-tight max-w-4xl">
          <span style={{ color: "#2ecc71" }}>FGSM</span>
          <br />
          <span className="text-white/90">Adversarial Attack</span>
          <br />
          <span
            className="text-transparent bg-clip-text animate-gradient"
            style={{
              backgroundImage: "linear-gradient(135deg, #2ecc71, #6be3a0, #2ecc71)",
              backgroundSize: "200% 200%",
            }}
          >
            Visualizer
          </span>
        </h1>

        {/* Subtitle */}
        <p
          className="animate-fade-up-d2 mt-6 text-lg md:text-xl max-w-2xl leading-relaxed"
          style={{ color: "#7ecda0" }}
        >
          Explore how the <strong style={{ color: "#2ecc71" }}>Fast Gradient Sign Method</strong> can
          fool neural networks with imperceptible perturbations.
          Upload an image and watch AI deception in real time.
        </p>

        {/* FGSM Formula */}
        <div
          className="animate-fade-up-d3 mt-8 rounded-xl px-6 py-3 font-mono text-sm md:text-base border"
          style={{
            color: "#2ecc71",
            borderColor: "rgba(46,204,113,0.2)",
            background: "rgba(46,204,113,0.06)",
          }}
        >
          x<sub>adv</sub> = x + &epsilon; &middot; sign(&nabla;<sub>x</sub> J(&theta;, x, y))
        </div>

        {/* CTA Buttons */}
        <div className="animate-fade-up-d4 flex flex-col sm:flex-row gap-4 mt-10">
          <button
            onClick={() => router.push("/attack")}
            className="btn-primary rounded-xl px-10 py-4 text-base font-semibold text-white cursor-pointer"
          >
            Get Started
          </button>
          <a
            href="https://arxiv.org/abs/1412.6572"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-outline rounded-xl px-10 py-4 text-base font-semibold cursor-pointer text-center"
          >
            Read the Paper
          </a>
        </div>

        {/* Feature Cards */}
        <div className="animate-fade-up-d5 grid grid-cols-1 sm:grid-cols-3 gap-5 mt-16 max-w-3xl w-full">
          {[
            {
              icon: "🧠",
              title: "MNIST Model",
              desc: "Pretrained CNN classifier for handwritten digits",
            },
            {
              icon: "⚡",
              title: "Real-time Attack",
              desc: "Upload an image and see FGSM in action instantly",
            },
            {
              icon: "📊",
              title: "Visual Comparison",
              desc: "Side-by-side original vs adversarial results",
            },
          ].map((card, i) => (
            <div
              key={i}
              className="card-hover rounded-xl p-5 text-center border"
              style={{
                borderColor: "#1a3526",
                background: "#111e16",
              }}
            >
              <div className="text-3xl mb-3 animate-float" style={{ animationDelay: `${i * 0.5}s` }}>
                {card.icon}
              </div>
              <h3 className="font-semibold text-sm mb-1" style={{ color: "#2ecc71" }}>
                {card.title}
              </h3>
              <p className="text-xs" style={{ color: "#7ecda0" }}>
                {card.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Footer */}
        <p className="animate-fade-up-d5 mt-16 text-xs" style={{ color: "#4a7a5e" }}>
          Built with Next.js + FastAPI + PyTorch &middot; Goodfellow et al., 2014
        </p>
      </div>
    </div>
  );
}
