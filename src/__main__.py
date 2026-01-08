import pandas as pd
import matplotlib.pyplot as plt
import math


def gradient_descent(m_now, b_now, points, L):
	m_gradient = 0
	b_gradient = 0
	n = len(points)

	for i in range(n):
		x = points.iloc[i].weight
		y = points.iloc[i].mpg

		y_pred = m_now * x + b_now
		err = y - y_pred

		m_gradient += -(2 / n) * x * err
		b_gradient += -(2 / n) * err

		if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(m_now) and math.isfinite(b_now) and math.isfinite(err)):
			print("DIVERGED at i =", i)
			print("x,y,m,b,err =", x, y, m_now, b_now, err)
			raise SystemExit


	m = m_now - m_gradient * L
	b = b_now - b_gradient * L

	if not (math.isfinite(m) and math.isfinite(b)):
		print("PARAMS DIVERGED")
		print("m,b =", m, b)
		raise SystemExit

	return m, b


def main():
	m = 0
	b = 0
	L = 0.01
	epochs = 1000
	
	data = pd.read_csv("resources/auto-mpg.csv")
	
	# Validate the data, make it all numeric and drop NaN values
	data = data[["weight", "mpg"]].copy()
	data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
	data["mpg"] = pd.to_numeric(data["mpg"], errors="coerce")
	data = data.dropna()

	# Get the mean/mittelwert and the standard deviation / standardabweichung
	w_mean, w_std = data["weight"].mean(), data["weight"].std()
	m_mean, m_std = data["mpg"].mean(), data["mpg"].std()

	# Apply the normalized values to the pandas dataframe
	data["weight"] = (data["weight"] - w_mean) / w_std
	data["mpg"] = (data["mpg"] - m_mean) / m_std

	print("weight mean/std:", data["weight"].mean(), data["weight"].std())
	print("mpg mean/std:", data["mpg"].mean(), data["mpg"].std())

	for i in range(epochs):
		if (i % 50 == 0):
			print(f"epoch {i}") # Print every 50 epochs
		m, b = gradient_descent(m, b, data, L)

	print(m, b)

	print("MODELING DONE!")
	print("Now testing the model")

	# raw input
	weight_lbs = 3968

	# normalize input
	x_norm = (weight_lbs - w_mean) / w_std

	# predict in normalized mpg space
	y_norm = m * x_norm + b

	# un-normalize mpg
	mpg_pred = y_norm * m_std + m_mean

	print(f"Predicted MPG for {weight_lbs} lbs: {mpg_pred:.2f}")

	plt.scatter(data.weight, data.mpg)
	xs = data.weight.sort_values()
	plt.plot(xs, m * xs + b, color="red")
	plt.show()


if __name__ == "__main__":
	main()





