# import unittest
# import numpy as np
#
# from ESN_handle import ESN
#
# N_in, N, N_out = 5, 75, 3
#
#
# def random_task():
#     X = np.random.randn(100, N_in)
#     y = np.random.randn(100, N_out)
#     Xp = np.random.randn(50, N_in)
#     return X, y, Xp
#
# class Performance(unittest.TestCase):
#     # Slighty bending the concept of a unit test, I want to catch performance changes during refactoring.
#     # Ideally, this will expand to a collection of known tasks.
#
#     def test_mackey(self):
#         try:
#             data = np.load('mackey_glass_t17.npy')
#         except IOError:
#             self.skipTest("missing data")
#
#         esn = ESN(n_inputs=1,
#                   n_outputs=1,
#                   n_reservoir=1000,
#                   spectral_radius=0.9,
#                   random_state=42)
#
#         trainlen = 2000
#         future = 2000
#         esn.fit(np.ones(trainlen), data[:trainlen])
#         prediction = esn.predict(np.ones(future))
#         error = np.sqrt(
#             np.mean((prediction.flatten() - data[trainlen:trainlen + future])**2))
#         # self.assertAlmostEqual(error, 0.1396039098653574)
#
#     def test_freqgen(self):
#         rng = np.random.RandomState(42)
#
#         def frequency_generator(N, min_period, max_period, n_changepoints):
#             """returns a random step function + a sine wave signal that
#                changes its frequency at each such step."""
#             # vector of random indices < N, padded with 0 and N at the ends:
#             changepoints = np.insert(np.sort(rng.randint(0, N, n_changepoints)), [
#                                      0, n_changepoints], [0, N])
#             # list of interval boundaries between which the control sequence
#             # should be constant:
#             const_intervals = list(
#                 zip(changepoints, np.roll(changepoints, -1)))[:-1]
#             # populate a control sequence
#             frequency_control = np.zeros((N, 1))
#             for (t0, t1) in const_intervals:
#                 frequency_control[t0:t1] = rng.rand()
#             periods = frequency_control * \
#                 (max_period - min_period) + max_period
#
#             # run time through a sine, while changing the period length
#             frequency_output = np.zeros((N, 1))
#             z = 0
#             for i in range(N):
#                 z = z + 2 * np.pi / periods[i]
#                 frequency_output[i] = (np.sin(z) + 1) / 2
#             return np.hstack([np.ones((N, 1)), 1 - frequency_control]), frequency_output
#
#         N = 15000
#         min_period = 2
#         max_period = 10
#         n_changepoints = int(N / 200)
#         frequency_control, frequency_output = frequency_generator(
#             N, min_period, max_period, n_changepoints)
#
#         traintest_cutoff = int(np.ceil(0.7 * N))
#         train_ctrl, train_output = frequency_control[
#             :traintest_cutoff], frequency_output[:traintest_cutoff]
#         test_ctrl, test_output = frequency_control[
#             traintest_cutoff:], frequency_output[traintest_cutoff:]
#
#         esn = ESN(n_inputs=2,
#                   n_outputs=1,
#                   n_reservoir=200,
#                   spectral_radius=0.25,
#                   sparsity=0.95,
#                   noise=0.001,
#                   input_shift=[0, 0],
#                   input_scaling=[0.01, 3],
#                   teacher_scaling=1.12,
#                   teacher_shift=-0.7,
#                   out_activation=np.tanh,
#                   inverse_out_activation=np.arctanh,
#                   random_state=rng,
#                   silent=True)
#
#         pred_train = esn.fit(train_ctrl, train_output)
#         print("test error:")
#         pred_test = esn.predict(test_ctrl)
#         error = np.sqrt(np.mean((pred_test - test_output)**2))
#         print(error)
#         # self.assertAlmostEqual(error, 0.30519018985725715)
#
#
# if __name__ == '__main__':
#     unittest.main()

import numpy as np
from ESN_handle import ESN
from matplotlib import pyplot as plt
#%matplotlib inline

data = np.load('mackey_glass_t17.npy') #  http://minds.jacobs-university.de/mantas/code
esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 500,
          spectral_radius = 1.5,
          random_state=42)

trainlen = 2000
future = 2000
pred_training = esn.fit(np.ones(trainlen),data[:trainlen])

prediction = esn.predict(np.ones(future))
# print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))))
print("test error rate: \n"+str(np.mean(np.sqrt((prediction.flatten() - data[trainlen:trainlen+future])**2) / data[trainlen:trainlen+future])))

plt.figure(figsize=(11,5))
plt.plot(range(0,trainlen+future),data[0:trainlen+future],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.show()