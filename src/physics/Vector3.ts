/**
 * 3D Vector class for physics calculations
 * Uses right-handed coordinate system: X=forward, Y=right, Z=down (aircraft body frame)
 */
export class Vector3 {
  constructor(
    public x: number = 0,
    public y: number = 0,
    public z: number = 0
  ) {}

  // Factory methods
  static zero(): Vector3 {
    return new Vector3(0, 0, 0);
  }

  static up(): Vector3 {
    return new Vector3(0, 0, -1);
  }

  static forward(): Vector3 {
    return new Vector3(1, 0, 0);
  }

  static right(): Vector3 {
    return new Vector3(0, 1, 0);
  }

  // Basic operations
  add(v: Vector3): Vector3 {
    return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
  }

  subtract(v: Vector3): Vector3 {
    return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
  }

  multiply(scalar: number): Vector3 {
    return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  divide(scalar: number): Vector3 {
    return new Vector3(this.x / scalar, this.y / scalar, this.z / scalar);
  }

  // Vector operations
  dot(v: Vector3): number {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v: Vector3): Vector3 {
    return new Vector3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  magnitude(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  magnitudeSquared(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  normalized(): Vector3 {
    const mag = this.magnitude();
    if (mag < 1e-10) return Vector3.zero();
    return this.divide(mag);
  }

  negate(): Vector3 {
    return new Vector3(-this.x, -this.y, -this.z);
  }

  // Utility
  clone(): Vector3 {
    return new Vector3(this.x, this.y, this.z);
  }

  toString(): string {
    return `(${this.x.toFixed(3)}, ${this.y.toFixed(3)}, ${this.z.toFixed(3)})`;
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  // Distance and angle
  distanceTo(v: Vector3): number {
    return this.subtract(v).magnitude();
  }

  angleTo(v: Vector3): number {
    const dot = this.dot(v);
    const mag = this.magnitude() * v.magnitude();
    if (mag < 1e-10) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / mag)));
  }

  // Component-wise operations
  componentMultiply(v: Vector3): Vector3 {
    return new Vector3(this.x * v.x, this.y * v.y, this.z * v.z);
  }

  // Lerp
  lerp(v: Vector3, t: number): Vector3 {
    return this.add(v.subtract(this).multiply(t));
  }
}
