#ifndef VEC3_HPP_5710DADF_17EF_453C_A9C8_4A73DC66B1CD
#define VEC3_HPP_5710DADF_17EF_453C_A9C8_4A73DC66B1CD

#include <cmath>
#include <cassert>
#include <cstdlib>

struct Vec3f
{
	float x, y, z;

	float& operator[] (std::size_t aI) 
	{
		assert(aI < 3);
		return aI[&x]; // This is a bit sketchy, but concise and efficient.
	}

	float operator[] (std::size_t aI) const
	{
		assert(aI < 3);
		return aI[&x]; // This is a bit sketchy.
	}
};


//__host__ __device__
inline
Vec3f operator+(Vec3f aVec)
{
	return aVec;
}

__host__ __device__
inline
Vec3f operator-(Vec3f aVec) 
{
	return { -aVec.x, -aVec.y, -aVec.z };
}

__host__ __device__
inline
Vec3f operator+(Vec3f aLeft, Vec3f aRight)
{
	return Vec3f{
		aLeft.x + aRight.x,
		aLeft.y + aRight.y,
		aLeft.z + aRight.z
	};
}

__host__ __device__
inline
Vec3f operator-(Vec3f aLeft, Vec3f aRight)
{
	return Vec3f{
		aLeft.x - aRight.x,
		aLeft.y - aRight.y,
		aLeft.z - aRight.z
	};
}

__host__ __device__
inline
Vec3f operator*(float aScalar, Vec3f aVec)
{
	return Vec3f{
		aScalar * aVec.x,
		aScalar * aVec.y,
		aScalar * aVec.z
	};
}

__host__ __device__
inline
Vec3f operator*(Vec3f aVec, float aScalar)
{
	return aScalar * aVec;
}

//__host__ __device__
inline
Vec3f operator/(Vec3f aVec, float aScalar)
{
	return Vec3f{
		aVec.x / aScalar,
		aVec.y / aScalar,
		aVec.z / aScalar
	};
}

//__host__ __device__
inline
Vec3f& operator+=(Vec3f& aLeft, Vec3f aRight)
{
	aLeft.x += aRight.x;
	aLeft.y += aRight.y;
	aLeft.z += aRight.z;
	return aLeft;
}

//__host__ __device__
inline
Vec3f& operator-=(Vec3f& aLeft, Vec3f aRight)
{
	aLeft.x -= aRight.x;
	aLeft.y -= aRight.y;
	aLeft.z -= aRight.z;
	return aLeft;
}

//__host__ __device__
inline
Vec3f& operator*=(Vec3f& aLeft, float aRight)
{
	aLeft.x *= aRight;
	aLeft.y *= aRight;
	aLeft.z *= aRight;
	return aLeft;
}

//__host__ __device__
inline
Vec3f& operator/=(Vec3f& aLeft, float aRight) 
{
	aLeft.x /= aRight;
	aLeft.y /= aRight;
	aLeft.z /= aRight;
	return aLeft;
}


// A few common functions:
__host__ __device__
inline
Vec3f cross(Vec3f aLeft, Vec3f aRight) 
{
	return {
		(aLeft.y * aRight.z) - (aLeft.z * aRight.y),
		(aLeft.z * aRight.x) - (aLeft.x * aRight.z),
		(aLeft.x * aRight.y) - (aLeft.y * aRight.x)
	};
}

__host__ __device__
inline
float dot(Vec3f aLeft, Vec3f aRight)
{
	return aLeft.x * aRight.x
		+ aLeft.y * aRight.y
		+ aLeft.z * aRight.z
		;
}

__host__ __device__
inline
float magnitude(Vec3f v) {
	float magnitude = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
	return magnitude;
}

//__host__ __device__
inline
Vec3f unit(Vec3f v) {
	float magnitude = sqrt(pow(v.x,2) + pow(v.y,2) + pow(v.z,2));
	return Vec3f{ v.x / magnitude, v.y / magnitude, v.z / magnitude };
}

//__host__ __device__
inline
float length(Vec3f aVec)
{
	// The standard function std::sqrt() is not marked as constexpr. length()
	// calls std::sqrt() unconditionally, so length() cannot be marked
	// constexpr itself.
	return std::sqrt(dot(aVec, aVec));
}

#endif