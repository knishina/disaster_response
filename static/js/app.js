function plotPie() {
    var default_url = "/genres";
    Plotly.d3.json(default_url, function(error, response) {
        if (error) return console.warn(error);
        var labels = Object.keys(response);
        var values = Object.values(response);
        
        var trace1 = {
            values: values,
            labels: labels,
            type: "pie",
        };
        var layout1 = {
            title: "Distribution of Message Genres"
        };
        var data = [trace1];
        Plotly.newPlot("pie", data, layout1)
    });
};
plotPie();


function plotBar() {
    var default_url = "/categories";
    Plotly.d3.json(default_url, function(error, response) {
        if (error) return console.warn(error);
        var labels = Object.keys(response);
        var values = Object.values(response);

        var trace1 = {
            x: labels,
            y: values, 
            type: "bar"
        };
        var layout1 = {
            title: "Distribution of Message Categories",
            xaxis: {
                title: "Categories",
                tickangle: -30,
                
            },
            yaxis: {title: "Number of Messages"}
        };
        var data = [trace1];
        Plotly.newPlot("plot", data, layout1)
    });
};
plotBar();


// function getTable() {
//     var default_url = "/text";
//     var tabled = Object.d3.json(default_url, function(error, response){
//         if (error) return console.warn(error);
//             console.log(response);
//             d3.select("tbody")
//                 .selectAll("tr")
//                 .data(response)
//                 .enter()
//                 .append("tr")
//                 .html(function(d) {
//             return `<td>${d.categories}</td>`
//         })
//     }); 
//     return tabled
// }

// getTable();