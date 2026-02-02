'use client';

import { useEffect, useRef } from 'react';

export default function NetworkBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        const particles: Particle[] = [];
        const particleCount = Math.min(Math.floor(width * 0.1), 100); // Responsive count
        const connectionDistance = 150;
        const mouseDistance = 200;

        let mouseX = 0;
        let mouseY = 0;

        class Particle {
            x: number;
            y: number;
            vx: number;
            vy: number;
            size: number;

            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.size = Math.random() * 2 + 1;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                // Bounce off edges
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;

                // Mouse interaction
                const dx = mouseX - this.x;
                const dy = mouseY - this.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < mouseDistance) {
                    const angle = Math.atan2(dy, dx);
                    const force = (mouseDistance - distance) / mouseDistance;
                    const pushX = Math.cos(angle) * force * 0.5;
                    const pushY = Math.sin(angle) * force * 0.5;

                    this.x -= pushX;
                    this.y -= pushY;
                }
            }

            draw() {
                if (!ctx) return;
                ctx.fillStyle = 'rgba(16, 185, 129, 0.5)'; // Emerald green
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Initialize particles
        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        const handleResize = () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        };

        const handleMouseMove = (e: MouseEvent) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
        };

        const animate = () => {
            if (!ctx) return;
            ctx.clearRect(0, 0, width, height);

            // Update and draw particles
            particles.forEach(p => {
                p.update();
                p.draw();
            });

            // Draw connections
            connectParticles();

            requestAnimationFrame(animate);
        };

        const connectParticles = () => {
            if (!ctx) return;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < connectionDistance) {
                        const opacity = 1 - (distance / connectionDistance);
                        ctx.strokeStyle = `rgba(16, 185, 129, ${opacity * 0.2})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        };

        window.addEventListener('resize', handleResize);
        window.addEventListener('mousemove', handleMouseMove);
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed top-0 left-0 w-full h-full pointer-events-none -z-10"
            style={{ background: 'radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%)' }}
        />
    );
}
