var temps = {};


// Define historic climate data

temps = {
	'Bangor'			:
		{
			thiswinter_low:	-20, 	thiswinter_lowdate: 'Jan. 4',
			record_low: -32, 	record_lowdate: 'Feb. 2, 1948', record_lowsince: 1925
		},
	'Caribou'			:
		{
			thiswinter_low:	-28, 	thiswinter_lowdate: 'Jan. 2',
			record_low: -41, record_lowdate: 'Feb. 1, 1955', record_lowsince: 1939
		},
	'Presque Isle':
		{
			thiswinter_low:	-29, 	thiswinter_lowdate: 'Jan. 2, Jan. 4',
			record_low: -41, record_lowdate: 'Jan. 19, 1925', record_lowsince: 1893
		},
	'Houlton'			:
		{
			thiswinter_low:	-24, 	thiswinter_lowdate: 'Jan. 2',
			record_low: -41, record_lowdate: 'Jan. 4, 1981', record_lowsince: 1948
		},
	'Millinocket'	:
		{
			thiswinter_low:	-16, 	thiswinter_lowdate: 'Jan. 1, Jan. 2',
			record_low: -40, record_lowdate: 'Feb. 2, 1962', record_lowsince: 1944
		},
	'Waterville'	:
		{
			thiswinter_low:	-18,	thiswinter_lowdate: 'Jan. 4', record_low: -39, record_lowdate: 'Dec. 30, 1933', record_lowsince: 1896
		},
	'Bar Harbor'	:
		{
			thiswinter_low:	-9, 	thiswinter_lowdate: 'Jan. 4'
		}	,
	'Sanford'			:
		{
			thiswinter_low:	-22,	thiswinter_lowdate: 'Jan. 4'
		}	,
	'Portland'		:
		{
			thiswinter_low:	-14,	thiswinter_lowdate: 'Jan. 4',
			record_low: -39, record_lowdate: 'Feb. 16, 1943', record_lowsince: 1874
		},
	'Augusta'			:
		{
			thiswinter_low:	-13,	thiswinter_lowdate: 'Jan. 4', record_low: -23, record_lowdate: 'Feb. 2, 1962', record_lowsince: 1948
		},
	'Auburn'			:
		{
			thiswinter_low:	-18,	thiswinter_lowdate: 'Jan. 4'
		}
};



// Definte current temperature data (can be pulled via a php script from weather.gov)

temps['Sanford'].currenttemp 		  = 11	;
temps['Portland'].currenttemp 		=	10	;
temps['Augusta'].currenttemp			=	-2	;
temps['Bar Harbor'].currenttemp	  =	12	;
temps['Auburn'].currenttemp			  =	1		;
temps['Waterville'].currenttemp	  = -15	;
temps['Millinocket'].currenttemp	= -10	;
temps['Houlton'].currenttemp			=	-3	;
temps['Caribou'].currenttemp			=	-20	;
temps['Presque Isle'].currenttemp =	-9	;
temps['Bangor'].currenttemp			  =	-19	;



$(document).ready(function() {

// Build the map

var width = 700,
    height = 800;

var projection = d3.geo.conicConformal()
    .parallels([38 + 02 / 60, 39 + 12 / 60])
    .rotate([70, 0])
    .scale(8500)
    .translate([0.4*width, 9.5*height]);

var path = d3.geo.path()
    .projection(projection)
    .pointRadius(30);

var svg = d3.select("#mainmap").append("svg")
    .attr("width", width)
    .attr("height",	 height);

var g = svg.append("g");

var colorrange = d3.scale.linear()
   .domain([-30, 50])
   .range(["rgb(0,191,255)", "rgb(255,0,0)"]);


d3.json("mainecitiesandcounties_topo.json", function(error, topo) {

// Append temperature data to data object
for (city in temps) {
	for (var i=0,len=topo.objects.cities.geometries.length; i<len; i++) {
		if (topo.objects.cities.geometries[i]['properties']['name'] === city) {
			topo.objects.cities.geometries[i]['properties']['temps'] = temps[city]
		}
	}
}



var counties = g.append("g")
		.attr("class", "counties")
  .selectAll("path")
    .data(topojson.feature(topo, topo.objects.counties).features)
	.enter().append("path")
		.attr("d", path);

var cities =
	g.append("g")
	.selectAll("path")
		.data(topojson.feature(topo, topo.objects.cities).features)
	.enter().append("path")
		.attr("class", "place")
		.attr("d", path)
		.attr("centroid-x", function(d) {
			var centroid = path.centroid(d);
			return centroid[0];
		} )
		.attr("centroid-y", function(d) {
			var centroid = path.centroid(d);
			return centroid[1];
		});


var citylabels =
 g.append("g").selectAll(".currenttemperature-label")
    .data(topojson.feature(topo, topo.objects.cities).features)
  .enter().append("text")
    .attr("class", "currenttemperature-label")
    .attr("transform", function(d) { return "translate(" + projection(d.geometry.coordinates) + ")"; })
    .attr("dy", ".35em")
    .attr("dx", "-.5em")
    .text(function(d) {
    	if (d.properties.temps)	return d.properties.temps.currenttemp;
    })
    .style("fill", function(d) {
    	if (d.properties.temps) return colorrange(d.properties.temps.currenttemp);
    })
		.on("mouseover", function(d) {
			$('.currenttemperature-label').not(this).animate({opacity:0.1},200);
			var s = d.properties.name;
			var id = "#" + s.split(' ');
			console.log(id);
			$(id).fadeIn(500);
		})
		.on("mouseout", function(d) {
			$('.currenttemperature-label').animate({opacity:1},100);
    	var s = d.properties.name;
    	var id = "#" + s.split(' ');
			$(id).fadeOut(500);
		});

var annotations =
	g.append("g").selectAll(".more")
		.data(topojson.feature(topo, topo.objects.cities).features)
  .enter().append("text")
  	.text(function(d) {
  		return d.properties.name;
  	})
    .attr("class", "more")
    .attr("id", function(d) {
    	var s = d.properties.name;
    	return s.split(" ")[0];
  	})
    .attr("transform", function(d) { return "translate(" + projection(d.geometry.coordinates) + ")"; })
    .attr("dy", "2em")
    .attr("dx", "-.5em");

annotations.append('tspan')
	.text(function(d) {
		if (d.properties.temps && d.properties.temps.thiswinter_low) {
			return ("This winter's record low: ");
		}
	})
	.attr("x","-.5em").attr("dy","1em")
	.append('tspan').text(function(d)	{
		if (d.properties.temps && d.properties.temps.thiswinter_low) {
		return d.properties.temps.thiswinter_low;
		}
	})
	.attr('class','number')
	.style("fill", function(d) {
		if (d.properties.temps && d.properties.temps.thiswinter_low) {
			return colorrange(d.properties.temps.thiswinter_low);
		}
	});


annotations.append("tspan").text(function(d) {
		if (d.properties.temps && d.properties.temps.thiswinter_low) {
				return "on " +  d.properties.temps.thiswinter_lowdate;
			}
    }).attr("x","-.5em").attr("dy","1em");

annotations.append("tspan").text(function(d) {
		if (d.properties.temps && d.properties.temps.record_low) {
				return "Record low (since " +  d.properties.temps.record_lowsince + "): ";
			}
    }).attr("x","-.5em").attr("dy","1.1em")
    .append('tspan').text(function(d) {
    	if (d.properties.temps && d.properties.temps.record_low) {
    	 return d.properties.temps.record_low;
    	}
   	})
   	.attr('class','number')
   	.style("fill", function(d) {
    	if (d.properties.temps && d.properties.temps.record_low) { return colorrange(d.properties.temps.record_low); }
    });

annotations.append("tspan").text(function(d) {
			if (d.properties.temps  && d.properties.temps.record_low) {
				return "on " +  d.properties.temps.record_lowdate;
			}
		})
		.attr("x","-.5em").attr("dy","1em")


});

});
