#This programme calculate area for given objects

class Circle:
  pi = 3.14
  def area(self, radius):
    area = Circle.pi * radius ** 2
    return area

circle = Circle()

pizza_area = circle.area(6)
teaching_table_area = circle.area(18)
round_room_area = circle.area(5730)
