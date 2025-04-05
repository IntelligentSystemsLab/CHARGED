# Data Description

Charging data and related auxiliary data and statistics for six cities: Los Angeles, São Paulo, Shenzhen, Amsterdam, Johannesburg, and Melbourne.

- There are two versions for each city, in which xxx_remove_zero removes charging stations with zero charge volume, charge duration and price.

- There are 10 CSV files in each folder, some of them are compressed in csv-larger-than-100MB.rar due to size limitation.

## distance.csv

- A distance matrix constructed based on the geographical coordinates (latitude and longitude) of charging stations.

- The matrix is computed using a geodesic algorithm based on the ellipsoidal Earth model.

- Both row and column indices defined by the unique identifiers of the aggregated stations, resulting in a symmetric matrix.

- With unit in kilometer.

## duration.csv

- Charging duration.

- The row indices represent a time series at an hourly granularity, and the column indices correspond to the station ID.

- With unit in hour(h).

## e_price.csv

- Charging fee.

- The row indices represent a time series at an hourly granularity, and the column indices correspond to the station ID.

- The units vary by city:

    - Los Angeles: USD/kWh

    - São Paulo: Brazilian Real/kWh

    - Shenzhen: CNY/kWh

    - Amsterdam: Dutch Guilder/kWh

    - Johannesburg: Rand/kWh

    - Melbourne: AUD/kWh

## info.csv

- Information about the data.

- The following table details the specific fields, their meanings, and corresponding units (if applicable) for the CSV file:

<table style="margin: 0 auto; table-layout: fixed;">
   <thead>
    <tr>
      <th style="text-align:center;">Field</th>
      <th style="text-align:center;">Meaning</th>
      <th style="text-align:center;">Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">city</td>
      <td style="text-align:left;">The name of the city</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">country</td>
      <td style="text-align:left;">The country to which the city belongs</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">abbreviation</td>
      <td style="text-align:left;">The abbreviation of the city</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_piles</td>
      <td style="text-align:left;">The total number of charging piles in the city</td>
      <td style="text-align:center;">Count</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_stations</td>
      <td style="text-align:left;">The total number of charging stations in the city</td>
      <td style="text-align:center;">Count</td>
    </tr>
    <tr>
      <td style="text-align:center;">DBSCAN_eps</td>
      <td style="text-align:left;">The epsilon parameter used for DBSCAN clustering of charging stations within the city, representing the neighborhood radius</td>
      <td style="text-align:center;">Meter (m)</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_duration</td>
      <td style="text-align:left;">The total charging duration recorded in the city</td>
      <td style="text-align:center;">Hour (h)</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_volume</td>
      <td style="text-align:left;">The total charging volume recorded in the city</td>
      <td style="text-align:center;">Kilowatt-hours (kWh)</td>
    </tr>
    <tr>
      <td style="text-align:center;">avg_power</td>
      <td style="text-align:left;">The average charging power in the city</td>
      <td style="text-align:center;">Kilowatts (kW)</td>
    </tr>
  </tbody>
</table>

## piles.csv

- Information about the piles.

- The following table details the specific fields, their meanings, and corresponding units (if applicable) for the CSV file:

<table style="margin: 0 auto; table-layout: fixed;">
   <thead>
    <tr>
      <th style="text-align:center;">Field</th>
      <th style="text-align:center;">Meaning</th>
      <th style="text-align:center;">Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">pile_id</td>
      <td style="text-align:left;">Unique identifier for the charging pile</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">longitude</td>
      <td style="text-align:left;">Geographical longitude of the charging pile's location</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">latitude</td>
      <td style="text-align:left;">Geographical latitude of the charging pile's location</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">station_id</td>
      <td style="text-align:left;">Identifier for the charging station to which the pile belongs</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_duration</td>
      <td style="text-align:left;">Total charging duration recorded at the charging pile</td>
      <td style="text-align:center;">Hours</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_volume</td>
      <td style="text-align:left;">Total amount of energy delivered at the charging pile</td>
      <td style="text-align:center;">Kilowatt-hours (kWh)</td>
    </tr>
    <tr>
      <td style="text-align:center;">avg_power</td>
      <td style="text-align:left;">Average power output during charging sessions</td>
      <td style="text-align:center;">Kilowatts (kW)</td>
    </tr>
  </tbody>
</table>

## poi.csv

- Urban Points of Interest (POIs) are obtained from the OpenStreetMap.

- The specific categories of POIs and their corresponding semantic definitions can be referenced in [OSM Wiki](https://wiki.openstreetmap.org/wiki/Map_features) or the associated data documentation.

- The following table details the specific fields, their meanings, and corresponding units (if applicable) for the CSV file:

<table style="margin: 0 auto; table-layout: fixed;">
   <thead>
    <tr>
      <th style="text-align:center;">Field</th>
      <th style="text-align:center;">Meaning</th>
      <th style="text-align:center;">Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">type</td>
      <td style="text-align:left;"> The geographic feature of the POI</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">longitude</td>
      <td style="text-align:left;">Geographical longitude of the POI's location</td>
      <td style="text-align:center;">Degrees</td>
    </tr>
    <tr>
      <td style="text-align:center;">latitude</td>
      <td style="text-align:left;">Geographical latitude of the POI's location</td>
      <td style="text-align:center;">Degrees</td>
    </tr>
  </tbody>
</table>



## s_price.csv

- Service fee.

- For São Paulo, Johannesburg, and Melbourne, no additional service fees are charged during the charging process; therefore, the service fee is uniformly assigned as 0.

- The row indices represent a time series at an hourly granularity, and the column indices correspond to the station ID.

- The units vary by city:

    - Los Angeles: USD/kWh

    - São Paulo: Brazilian Real/kWh

    - Shenzhen: CNY/kWh

    - Amsterdam: Dutch Guilder/kWh

    - Johannesburg: Rand/kWh

    - Melbourne: AUD/kWh


## stations.csv

- Information about the stations.

- The following table details the specific fields, their meanings, and corresponding units (if applicable) for the CSV file:

<table style="margin: 0 auto; table-layout: fixed;">
   <thead>
    <tr>
      <th style="text-align:center;">Field</th>
      <th style="text-align:center;">Meaning</th>
      <th style="text-align:center;">Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">station_id</td>
      <td style="text-align:left;">Unique identifier for the charging station</td>
      <td style="text-align:center;">N/A</td>
    </tr>
    <tr>
      <td style="text-align:center;">longitude</td>
      <td style="text-align:left;">Geographical longitude of the charging station's location</td>
      <td style="text-align:center;">Degrees</td>
    </tr>
    <tr>
      <td style="text-align:center;">latitude</td>
      <td style="text-align:left;">Geographical latitude of the charging station's location</td>
      <td style="text-align:center;">Degrees</td>
    </tr>
    <tr>
      <td style="text-align:center;">pile_num</td>
      <td style="text-align:left;">Number of charging piles at the station</td>
      <td style="text-align:center;">Count</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_duration</td>
      <td style="text-align:left;">Total charging duration recorded at the charging station</td>
      <td style="text-align:center;">Hours</td>
    </tr>
    <tr>
      <td style="text-align:center;">total_volume</td>
      <td style="text-align:left;">Total charging volume recorded at the charging station</td>
      <td style="text-align:center;">Kilowatt-hours (kWh)</td>
    </tr>
    <tr>
      <td style="text-align:center;">avg_power</td>
      <td style="text-align:left;">Average charging power at the charging station</td>
      <td style="text-align:center;">Kilowatts (kW)</td>
    </tr>
    <tr>
      <td style="text-align:center;">perimeter</td>
      <td style="text-align:left;">Perimeter of the shape formed after DBSCAN clustering</td>
      <td style="text-align:center;">Meters (m)</td>
    </tr>
    <tr>
      <td style="text-align:center;">area</td>
      <td style="text-align:left;">Area of the shape formed after DBSCAN clustering</td>
      <td style="text-align:center;">Square meters (m<sup>2</sup>)</td>
    </tr>
  </tbody>
</table>


## volume.csv

- Charging volume.

- The row indices represent a time series at an hourly granularity, and the column indices correspond to the station ID.

- With unit in Kilowatt-hours (kWh).


## weather.csv

- Hourly-granularity weather data obtained and organized through the API services provided by Visual Crossing.

- The following field explanations are based on the [documentation provided by Visual Crossing](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/), which details the various meteorological parameters and their associated attributes:

    - temp – temperature at the location.

    - feelslike – what the temperature feels like accounting for heat index or wind chill.

    - humidity – relative humidity in %.

    - dew – dew point temperature.

    - precip – the amount of liquid precipitation that fell or is predicted to fall in the period. This includes the liquid-equivalent amount of any frozen precipitation such as snow or ice.

    - snow – the amount of snow that fell or is predicted to fall.

    - snowdepth – the depth of snow on the ground.

    - preciptype – an array indicating the type(s) of precipitation expected or that occurred. Possible values include rain, snow, freezingrain and ice.

    - windgust  – instantaneous wind speed at a location. May be empty if it is not significantly higher than the wind speed.

    - windspeed – the sustained wind speed measured as the average windspeed that occurs during the preceding one to two minutes.

    - winddir – direction from which the wind is blowing.

    - pressure – the sea level atmospheric or barometric pressure in millibars (or hectopascals).

    - visibility – distance at which distant objects are visible.

    - cloudcover – how much of the sky is covered in cloud ranging from 0-100%.

    - solarradiation  – (W/m2) the solar radiation power at the instantaneous moment of the observation (or forecast prediction).

    - solarenergy  – (MJ/m2) indicates the total energy from the sun that builds up over an hour or day.

    - uvindex – a value between 0 and 10 indicating the level of ultra violet (UV) exposure for that hour or day. 10 represents high level of exposure, and 0 represents no exposure. The UV index is calculated based on amount of short wave solar radiation which in turn is a level the cloudiness, type of cloud, time of day, time of year and location altitude.

    - conditions – textual representation of the weather conditions. Conditions represented by numbers correspond to the following: 

        - 'Clear': 0,
        - 'Overcast': 1,
        - 'Partially cloudy': 2,
        - 'Rain': 3,
        - 'Rain, Fog': 4,
        - 'Rain, Overcast': 5,
        - 'Rain, Partially cloudy': 6,
        - 'Snow': 7,
        - 'Snow, Fog': 8,
        - 'Snow, Partially cloudy': 9,
        - 'Snow, Rain': 10,
        - 'Snow, Rain, Overcast': 11,
        - 'Snow, Rain, Partially cloudy': 12

