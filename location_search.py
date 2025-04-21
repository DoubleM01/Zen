import overpy
from gps3 import gps3
import math
from typing import List, Tuple, Dict, Any
from RouteCalculator import RouteCalculator
import os
LON = None
LAT = None

def get_gps_location():
    gps_socket = gps3.GPSDSocket()
    data_stream = gps3.DataStream()
    gps_socket.connect()
    gps_socket.watch()

    for new_data in gps_socket:
        if new_data:
            data_stream.unpack(new_data)
            LAT = data_stream.TPV['lat']
            LON = data_stream.TPV['lon']
            if LAT != 'n/a' and LON != 'n/a':
                return float(LAT), float(LON)
    raise RuntimeError("No GPS fix")

def build_overpass_query(lat: float, lon: float, radius: int) -> str:
    # We search for fitness centers, gyms, and sports centers
    # This query looks for amenities and leisure facilities that are typically gyms
    
    return f"""
    (
      node["leisure"="fitness_centre"](around:{radius},{lat},{lon});
      way["leisure"="fitness_centre"](around:{radius},{lat},{lon});
      relation["leisure"="fitness_centre"](around:{radius},{lat},{lon});
      
      node["amenity"="gym"](around:{radius},{lat},{lon});
      way["amenity"="gym"](around:{radius},{lat},{lon});
      relation["amenity"="gym"](around:{radius},{lat},{lon});
      
      node["sport"="fitness"](around:{radius},{lat},{lon});
      way["sport"="fitness"](around:{radius},{lat},{lon});
      relation["sport"="fitness"](around:{radius},{lat},{lon});
      
      node["leisure"="sports_centre"](around:{radius},{lat},{lon});
      way["leisure"="sports_centre"](around:{radius},{lat},{lon});
      relation["leisure"="sports_centre"](around:{radius},{lat},{lon});
    );
    out center;
    """

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, name) :
    """
    Calculate the distance between two points using the Haversine formula.
    
    Args:
        lat1 (float): Latitude of point 1
        lon1 (float): Longitude of point 1
        lat2 (float): Latitude of point 2
        lon2 (float): Longitude of point 2
        
    Returns:
        float: Distance in meters
    """
    distance = None
    time = None
    calculator = RouteCalculator()
    start_coords = (lat1, lon1)
    end_coords = (lat2, lon2)
    
    if start_coords and end_coords:
        result = calculator.calculate_route(start_coords, end_coords)
        
        if result:
            print(f"\nWalking Route Details:")
            print(f"Total distance: {calculator.format_distance(result['total_distance'])}")
            print(f"Estimated time: {calculator.format_duration(result['total_duration'])}")
            distance = calculator.format_distance(result['total_distance'])
            time = calculator.format_duration(result['total_duration'])
            # print("\nTurn-by-turn directions:")
            # for i, instruction in enumerate(result['instructions'], 1):
            #     distance = calculator.format_distance(instruction['distance'])
            #     print(f"{i}. {instruction['text']} - {distance}")
            
            # Create and save map
            map_obj = calculator.create_map(result, "your location", "GYM")
            if map_obj:
                map_obj.save(os.path.join("static", name + "_walking_route.html"))
                print("\nMap saved as 'walking_route.html'")
        else:
            print("Unable to calculate route")
    else:
        print("Unable to find locations")

    return distance , time
def estimate_walking_time(distance: float) -> int:
    """
    Estimate walking time based on distance.
    
    Args:
        distance (float): Distance in meters
        
    Returns:
        int: Estimated walking time in minutes
    """
    # Average walking speed is about 5 km/h or 1.4 m/s
    walking_speed = 1.4  # meters per second
    
    # Calculate time in seconds
    time_seconds = distance / walking_speed
    
    # Convert to minutes and round up
    time_minutes = math.ceil(time_seconds / 60)
    
    return time_minutes

def get_nearby_gyms(lat, lon, radius=3000):
    """
    Find nearby gyms using OpenStreetMap data.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius (int): Search radius in meters, default is 4000 (4km)
        
    Returns:
        list: List of tuples containing (name, latitude, longitude, distance, walking_time) of nearby gyms
    """
    gyms = []
    api = overpy.Overpass()
    
    try:
        query = build_overpass_query(lat, lon, radius)
        print(f"Searching for gyms near {lat}, {lon} with radius {radius}m")
        result = api.query(query)
        
        # Process nodes (points)
        for node in result.nodes:
            name = node.tags.get("name", "Unnamed Gym")
            gym_lat = float(node.lat)
            gym_lon = float(node.lon)
            distance,walking_time = calculate_distance(lat, lon, gym_lat, gym_lon, name)
            #walking_time = RouteCalculator.format_duration(estimate_walking_time(walking_time))
            gyms.append((name, gym_lat, gym_lon, distance, walking_time))
        
        # Process ways (areas)
        for way in result.ways:
            if way.center_lat and way.center_lon:
                name = way.tags.get("name", "Unnamed Gym")
                gym_lat = float(way.center_lat)
                gym_lon = float(way.center_lon)
                distance,walking_time = calculate_distance(lat, lon, gym_lat, gym_lon, name)
                #walking_time = RouteCalculator.format_duration(estimate_walking_time(walking_time))
                gyms.append((name, gym_lat, gym_lon, distance, walking_time))
        
        # Process relations (complex areas)
        for relation in result.relations:
            if hasattr(relation, 'center_lat') and hasattr(relation, 'center_lon'):
                if relation.center_lat and relation.center_lon:
                    name = relation.tags.get("name", "Unnamed Gym")
                    gym_lat = float(relation.center_lat)
                    gym_lon = float(relation.center_lon)
                    distance,walking_time = calculate_distance(lat, lon, gym_lat, gym_lon, name)
                    #walking_time = RouteCalculator.format_duration(estimate_walking_time(walking_time))
                    gyms.append((name, gym_lat, gym_lon, distance, walking_time))
        
        # Sort gyms by distance (closest first)
        gyms.sort(key=lambda x: x[3])
        
        print(f"Found {len(gyms)} gyms nearby")
        for name, gym_lat, gym_lon, distance, walking_time in gyms:
            print(f"- {name} at {gym_lat}, {gym_lon} - {distance} ({walking_time} min walk)")
            
    except Exception as e:
        print(f"Overpass error: {e}")
    
    return gyms

