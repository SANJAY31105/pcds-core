'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';
import { ThreatDetection } from '@/types';
import { Maximize2, Minimize2, Play, Pause } from 'lucide-react';

interface NetworkNode extends d3.SimulationNodeDatum {
    id: string;
    name: string;
    type: 'server' | 'client' | 'router' | 'threat';
    ip: string;
    riskScore: number;
    isActive: boolean;
    threats: ThreatDetection[];
}

interface NetworkLink {
    source: string | NetworkNode;
    target: string | NetworkNode;
    value: number;
    isThreat: boolean;
}

interface NetworkTopology3DProps {
    threats: ThreatDetection[];
}

export default function NetworkTopology3D({ threats }: NetworkTopology3DProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [nodeCount, setNodeCount] = useState(0);
    const simulationRef = useRef<d3.Simulation<NetworkNode, NetworkLink> | null>(null);

    useEffect(() => {
        if (!svgRef.current) return;

        const width = svgRef.current.clientWidth;
        const height = svgRef.current.clientHeight;

        // Clear previous content
        d3.select(svgRef.current).selectAll('*').remove();

        // Create SVG groups
        const svg = d3.select(svgRef.current);
        const g = svg.append('g');

        // Generate network topology from threats
        const { nodes, links } = generateNetworkTopology(threats);
        setNodeCount(nodes.length);

        // Create force simulation
        const simulation = d3.forceSimulation<NetworkNode, NetworkLink>(nodes)
            .force('link', d3.forceLink<NetworkNode, NetworkLink>(links)
                .id((d: any) => d.id)
                .distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        simulationRef.current = simulation;

        // Add zoom behavior
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom as any);

        // Create links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('class', 'network-link')
            .attr('stroke', (d: any) => d.isThreat ? '#ff0055' : 'rgba(0, 240, 255, 0.3)')
            .attr('stroke-width', (d: any) => d.isThreat ? 2 : 1)
            .attr('stroke-dasharray', (d: any) => d.isThreat ? '5,5' : '0');

        // Create node groups
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('class', 'network-node')
            .call(d3.drag<any, NetworkNode>()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }));

        // Add circles with gradient fills
        const defs = svg.append('defs');

        // Create gradients for different risk levels
        const createGradient = (id: string, color: string) => {
            const gradient = defs.append('radialGradient')
                .attr('id', id);
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', color)
                .attr('stop-opacity', 0.8);
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', color)
                .attr('stop-opacity', 0.2);
        };

        createGradient('gradient-critical', '#ff0055');
        createGradient('gradient-high', '#ff6b35');
        createGradient('gradient-medium', '#ffaa00');
        createGradient('gradient-low', '#00d4ff');
        createGradient('gradient-safe', '#00ff85');

        // Add circles
        node.append('circle')
            .attr('r', (d: any) => d.type === 'server' ? 20 : d.type === 'threat' ? 15 : 12)
            .attr('fill', (d: any) => {
                if (d.type === 'threat') return 'url(#gradient-critical)';
                if (d.riskScore > 80) return 'url(#gradient-critical)';
                if (d.riskScore > 60) return 'url(#gradient-high)';
                if (d.riskScore > 40) return 'url(#gradient-medium)';
                if (d.riskScore > 20) return 'url(#gradient-low)';
                return 'url(#gradient-safe)';
            })
            .attr('stroke', '#00f0ff')
            .attr('stroke-width', (d: any) => d.isActive ? 2 : 1)
            .attr('class', (d: any) => d.isActive ? 'node-active' : '');

        // Add glow effect for threat nodes
        node.filter((d: any) => d.type === 'threat' || d.riskScore > 70)
            .append('circle')
            .attr('r', (d: any) => d.type === 'server' ? 25 : d.type === 'threat' ? 20 : 17)
            .attr('fill', 'none')
            .attr('stroke', '#ff0055')
            .attr('stroke-width', 1)
            .attr('opacity', 0.5)
            .attr('class', 'node-glow');

        // Add labels
        node.append('text')
            .text((d: any) => d.name)
            .attr('font-size', 10)
            .attr('fill', '#ffffff')
            .attr('text-anchor', 'middle')
            .attr('dy', 30)
            .attr('class', 'node-label');

        // Add tooltips
        node.append('title')
            .text((d: any) => `${d.name}\nIP: ${d.ip}\nRisk: ${d.riskScore}/100\nThreats: ${d.threats.length}`);

        // Animation
        simulation.on('tick', () => {
            link
                .attr('x1', (d: any) => d.source.x)
                .attr('y1', (d: any) => d.source.y)
                .attr('x2', (d: any) => d.target.x)
                .attr('y2', (d: any) => d.target.y);

            node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
        });

        // Pulse animation for active nodes
        const pulse = () => {
            g.selectAll('.node-active')
                .transition()
                .duration(1000)
                .attr('r', 25)
                .transition()
                .duration(1000)
                .attr('r', 20);

            g.selectAll('.node-glow')
                .transition()
                .duration(1500)
                .attr('opacity', 0.8)
                .transition()
                .duration(1500)
                .attr('opacity', 0.3);
        };

        const pulseInterval = setInterval(pulse, 2000);

        return () => {
            clearInterval(pulseInterval);
            simulation.stop();
        };
    }, [threats, isPaused]);

    const generateNetworkTopology = (threats: ThreatDetection[]) => {
        const nodes: NetworkNode[] = [];
        const links: NetworkLink[] = [];
        const nodeMap = new Map<string, NetworkNode>();

        // Create central server
        const server: NetworkNode = {
            id: 'server-1',
            name: 'Main Server',
            type: 'server',
            ip: '10.0.0.1',
            riskScore: 25,
            isActive: true,
            threats: []
        };
        nodes.push(server);
        nodeMap.set(server.id, server);

        // Create router
        const router: NetworkNode = {
            id: 'router-1',
            name: 'Gateway',
            type: 'router',
            ip: '10.0.0.254',
            riskScore: 15,
            isActive: false,
            threats: []
        };
        nodes.push(router);
        nodeMap.set(router.id, router);

        // Add link between server and router
        links.push({
            source: server.id,
            target: router.id,
            value: 1,
            isThreat: false
        });

        // Create nodes from unique IPs in threats
        const ipSet = new Set<string>();
        threats.forEach(threat => {
            ipSet.add(threat.source_ip);
            if (threat.destination_ip) ipSet.add(threat.destination_ip);
        });

        const recentIPs = Array.from(ipSet).slice(0, 15);

        recentIPs.forEach((ip, idx) => {
            const relatedThreats = threats.filter(t =>
                t.source_ip === ip || t.destination_ip === ip
            );

            const avgRisk = relatedThreats.length > 0
                ? relatedThreats.reduce((sum, t) => sum + t.risk_score, 0) / relatedThreats.length
                : 20;

            const isThreatSource = relatedThreats.some(t => t.source_ip === ip && t.risk_score > 50);

            const node: NetworkNode = {
                id: `node-${ip}`,
                name: ip.split('.').slice(-2).join('.'),
                type: isThreatSource ? 'threat' : 'client',
                ip: ip,
                riskScore: avgRisk,
                isActive: relatedThreats.length > 0,
                threats: relatedThreats
            };

            nodes.push(node);
            nodeMap.set(node.id, node);

            // Connect to router or server
            const target = isThreatSource ? server.id : router.id;
            links.push({
                source: node.id,
                target: target,
                value: relatedThreats.length,
                isThreat: avgRisk > 60
            });
        });

        return { nodes, links };
    };

    const togglePause = () => {
        if (simulationRef.current) {
            if (isPaused) {
                simulationRef.current.restart();
            } else {
                simulationRef.current.stop();
            }
            setIsPaused(!isPaused);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`glass p-6 rounded-2xl ${isFullscreen ? 'fixed inset-4 z-50' : 'relative'}`}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h2 className="text-xl font-bold flex items-center">
                        <span className="text-cyber-blue mr-2">🌐</span>
                        Network Topology
                    </h2>
                    <p className="text-xs text-gray-400 mt-1">
                        {nodeCount} nodes • Real-time threat overlay
                    </p>
                </div>

                <div className="flex items-center space-x-2">
                    <button
                        onClick={togglePause}
                        className="glass px-3 py-2 rounded-lg hover:bg-white/10 transition-colors"
                        title={isPaused ? 'Resume' : 'Pause'}
                    >
                        {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                    </button>
                    <button
                        onClick={() => setIsFullscreen(!isFullscreen)}
                        className="glass px-3 py-2 rounded-lg hover:bg-white/10 transition-colors"
                        title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                    >
                        {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                    </button>
                </div>
            </div>

            {/* Legend */}
            <div className="flex items-center space-x-4 mb-4 text-xs">
                <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 rounded-full bg-threat-critical"></div>
                    <span className="text-gray-400">Critical</span>
                </div>
                <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 rounded-full bg-threat-high"></div>
                    <span className="text-gray-400">High</span>
                </div>
                <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 rounded-full bg-threat-medium"></div>
                    <span className="text-gray-400">Medium</span>
                </div>
                <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 rounded-full bg-cyber-green"></div>
                    <span className="text-gray-400">Safe</span>
                </div>
            </div>

            {/* 3D Network Visualization */}
            <div className={`bg-black/20 rounded-xl overflow-hidden ${isFullscreen ? 'h-[calc(100vh-200px)]' : 'h-[500px]'}`}>
                <svg
                    ref={svgRef}
                    className="w-full h-full"
                    style={{ background: 'radial-gradient(ellipse at center, rgba(0, 240, 255, 0.05) 0%, transparent 70%)' }}
                />
            </div>

            {/* Controls hint */}
            <div className="mt-3 text-xs text-gray-500 text-center">
                <span>💡 Drag nodes • Scroll to zoom • Hover for details</span>
            </div>

            <style jsx>{`
        .network-link {
          pointer-events: none;
        }
        .network-node {
          cursor: pointer;
        }
        .network-node:hover circle {
          stroke-width: 3;
        }
        .node-label {
          pointer-events: none;
          user-select: none;
        }
      `}</style>
        </motion.div>
    );
}
