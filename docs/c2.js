import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const margin = { top: 70, right: 30, bottom: 40, left: 80 };
const width = 1200 - margin.left - margin.right;
const height = 800 - margin.top - margin.bottom;

const raw_data = await d3.csv("generated/c2_experiment_data.csv");
const c2_data = raw_data.map(d => { return {
    c2: parseFloat(d.c2),
    gradient_norm: parseFloat(d.gradient_norm),
    step_index: parseInt(d.step_index),
}})

const x = d3.scaleLinear()
    .domain(d3.extent(c2_data, d => d.step_index)).nice()
    .range([margin.left, width - margin.right]);

// Define the vertical scale.
const y = d3.scaleLinear()
    .domain(d3.extent(c2_data, d => d.gradient_norm)).nice()
    .range([height - margin.bottom, margin.top]);

const svg = d3.select("#chart-container")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))

svg.append("text")
    .attr("x", margin.left + width / 2 - 100)
    .attr("y", height - margin.bottom + 50)
    .attr("text-anchor", "left")
    .text('number of iteration steps')

svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y))

svg.append("text")
    .attr("y", 30)
    .attr("x", -height / 2 - margin.top)
    .attr("text-anchor", "left")
    .attr("transform", "rotate(-90)")
    .text('gradient norm')

const line = d3.line()
    .x(d => x(d.step_index))
    .y(d => y(d.gradient_norm));

const c2_values = Array.from(d3.union(d3.map(c2_data, d => d.c2)))
const colorMap = d3.scaleOrdinal(c2_values, d3.schemeAccent);

c2_values.forEach(function(c2, c2_index, arr) {
    console.log(`c2=${c2}. c2_index=${c2_index}`)

    const c2_data_filtered = c2_data.filter(d => d.c2 == c2)
    const c2_color = colorMap(c2)

    svg.append("path")
        .datum(c2_data_filtered)
        .attr("fill", "none")
        .attr("stroke", c2_color)
        .attr("stroke-width", 1)
        .attr("d", line);

    // Legend circle.
    svg.append("circle")
        .attr("cx", width - 200)
        .attr("cy", 50 + margin.top + c2_index * 25)
        .attr("r", 7)
        .style("fill", c2_color)

    // Legend label.
    svg.append("text")
        .attr("x", width - 180)
        .attr("y", 50 + 7 + margin.top + c2_index * 25)
        .attr("text-anchor", "left")
        .text(`c2=${c2}`)
})
