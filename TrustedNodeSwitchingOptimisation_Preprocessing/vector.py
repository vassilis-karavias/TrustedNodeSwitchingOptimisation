import numpy as np
from copy import deepcopy

class Vector():

    def __init__(self, values):
        """
        Generate a vector class with components equal to the values in the array values
        :param values: The array with the elements of the vector in positions: numpy array
        """
        self.vector = values.flatten()

    def get_ith_element(self, i):
        """
        get the ith element of the vector
        :param i: The position in the array to set
        :return: The value of the ith element in the vector
        """
        # if the position of the element you are looking for is outside the range of the vector raise ValueError
        if i > len(self.vector):
            print("The element you are looking for is outside the range of the vector")
            raise ValueError
        else:
            return self.vector[i]

    def set_ith_element(self, i, new_element):
        """
        set the ith element of the vector to new value
        :param i: The position in the array to set
        :param new_element: The new value to set the element in the array
        """
        # if the value you wish to set is outside range of vector raise ValueError
        if i > len(self.vector):
            print("The element is outside the range of the vector")
            raise ValueError
        else:
            self.vector[i] = new_element

    def __add__(self, other):
        """
        carry out vector addition with self and other: v + u
        :param other: the other vector to add to this vector: Vector
        :return: The vector that is v + u
        """
        # if the lengths of the vectors are not equal then they cannot be added: raise ValueError
        if len(self.vector) != len(other.vector):
            print("Only vectors of the same length can be added")
            raise ValueError
        else:
            new_vector = []
            for i in range(len(self.vector)):
                new_vector.append(self.vector[i] + other.vector[i])
            return Vector(np.array(new_vector))

    def __sub__(self, other):
        """
        carry out vector subtraction with self and other: self - other
        :param other: The other vector to subtract from this vector
        :return: The vector self - other
        """
        # if the lengths of the vectors are not equal then they cannot be subtracted: raise ValueError
        if len(self.vector) != len(other.vector):
            print("Only vectors of the same length can be subtracted")
            raise ValueError
        else:
            new_vector = []
            for i in range(len(self.vector)):
                new_vector.append(self.vector[i] - other.vector[i])
            return Vector(np.array(new_vector))

    def scalar_mult(self, scalar):
        """
        carry out multiplication with a scalar value: scalar * vec
        :param scalar: The scalar to multiply the vector with
        :return: The vector that is = scalar * self
        """
        new_vector = []
        for i in range(len(self.vector)):
            new_vector.append(scalar * self.vector[i])
        return Vector(np.array(new_vector))

    def dot_prod(self, other):
        """
        carry out the dot product of two vectors self.other
        :param other: The other vector to carry out the dot product with
        :return: The result of the dot product: float
        """
        # if the length of the vectors are not the same then they cannot have dot product: raise ValueError
        if len(self.vector) != len(other.vector):
            print("Only vectors of the same length can be dot producted")
            raise ValueError
        else:
            dot_prod = 0.0
            for i in range(len(self.vector)):
                dot_prod += self.vector[i] * other.vector[i]
            return dot_prod

    def cross_prod(self, other):
        """
        carry out the cross product between self and other: self x other
        :param other: The other vector to carry out cross product with
        :return: The vector that is self x other
        """
        # cross product only defined for 3 dimensional vectors so raise ValueError if they are not 3 dimensional
        if len(self.vector) != len(other.vector) or len(self.vector) != 3:
            print("Cross product only defined for vectors of 3 dimensions")
            raise ValueError
        else:
            return Vector(np.array([self.vector[1] * other.vector[2] - self.vector[2] * other.vector[1],
                           self.vector[2] * other.vector[0] - self.vector[0] * other.vector[2],
                           self.vector[0] * other.vector[1] - self.vector[1] * other.vector[0]]))

    def magnitude(self):
        """
        find the magnitude of the vector |v|
        :return: The value of |v|: float
        """
        # |v| = (sum v_i^2) ^(1/2) = |v.v|^(1/2)
        magnitude = 0.0
        for i in range(len(self.vector)):
            magnitude += np.power(self.vector[i],2)
        return (magnitude ** 0.5)

    def normalise(self):
        """
        get a vector that is normalised
        :return: Normalised vector
        """
        magnitude = self.magnitude()
        if magnitude > 0.00000001:
            new_vector = Vector(deepcopy(self.vector))
            new_vector.scalar_mult(1/magnitude)
            return new_vector
        else:
            print("Zero Vector Cannot Be Normalised")
            return self