import matplotlib.pyplot as plt

# import re

# from preprocessing import gen_RT_matrix, get_rendered_convdata

# proj_dir = "."
# train_index_path = proj_dir + "/data/index/train.txt"

# train_index = []

# with open(train_index_path, "r") as f:
#     lines = f.readlines()
#     for x in lines:
#         train_index.append(x.strip().split(" "))


# def show3DhairPlot(index):
#     """
#     strands: [100, 4, 32, 32]
#     mask: [32, 32] bool
#     """
#     # f = open("hair.obj", "w+")

#     current_index = train_index[index]
#     current_convdata_index = re.search(
#         "strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d", str(current_index)
#     ).group(0)
#     current_RT_mat = gen_RT_matrix(proj_dir + "/data/" + str(current_index[0]) + ".txt")
#     current_convdata_path = (
#         proj_dir + "/convdata/" + str(current_convdata_index) + ".convdata"
#     )
#     # current_RT_mat = np.dot(current_RT_mat, gen_RT_matrix2([0, 0, 0, 0.5, 0.5, 0.5]))
#     strands = get_rendered_convdata(current_convdata_path, current_RT_mat)
#     show3DhairPlotByStrands(strands)

# def demo(weight_path, interp_factor, img_path):
#     print("This is the programme of demo.")
#     # load model
#     print("Building Network...")
#     net = Net()
#     # net.cuda()
#     net.load_state_dict(torch.load(weight_path, map_location="cpu"))
#     net.eval()

#     """ convdata_path = "convdata/" + img_path.split("/")[-1].split("_v1")[0] + ".convdata"
#     convdata = np.load(convdata_path).reshape(100, 4, 32, 32) """

#     # load demo data
#     img = cv2.imread(img_path)
#     img = gasuss_noise(img)
#     img = img.reshape(1, 3, 128, 128)
#     img = torch.from_numpy(img).float()

#     # img = img.cuda()
#     output = net(img, interp_factor)

#     strands = output[0].cpu().detach().numpy()  # hair strands

#     # axis-wise 1D gaussian filtering
#     gaussian = cv2.getGaussianKernel(10, 3)
#     for i in range(strands.shape[2]):
#         for j in range(strands.shape[3]):
#             strands[:, :3, i, j] = cv2.filter2D(strands[:, :3, i, j], -1, gaussian)

#     show3DhairPlotByStrands(strands)

#     # generate .data file (format in http://www-scf.usc.edu/~liwenhu/SHM/database.html)
#     """ epoch = weight_path[-16:-10]
#     save_binary(strands, "demo/epoch_{}.data".format(epoch))
#     ground_truth = convdata
#     save_binary(ground_truth, "demo/ground_truth.data") """


# def example(convdata):
#     strands = np.load(convdata).reshape(100, 4, 32, 32)
#     show3DhairPlotByStrands(strands)


def show3DhairPlotByStrands(strands):
    """
    strands: [100, 4, 32, 32]
    mask: [32, 32] bool
    """

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection="3d")

    avgx, avgy, avgz = 0, 0, 0

    count = 0
    for i in range(32):
        for j in range(32):
            if sum(sum(strands[:, :, i, j])) == 0:
                continue
            strand = strands[:, 0:3, i, j]
            # each strand now has shape (100, 3)
            x = strand[:, 0]
            y = strand[:, 1]
            z = strand[:, 2]
            ax.plot(x, y, z, linewidth=0.2, color="brown")

            avgx += sum(x) / 100
            avgy += sum(y) / 100
            avgz += sum(z) / 100
            count += 1

    avgx /= count
    avgy /= count
    avgz /= count

    RADIUS = 0.3  # space around the head
    ax.set_xlim3d([avgx - RADIUS, avgx + RADIUS])
    ax.set_ylim3d([avgy - RADIUS, avgy + RADIUS])
    ax.set_zlim3d([avgz - RADIUS, avgz + RADIUS])
    plt.show()


# show3DhairPlot(4)
# show3DhairPlot(200)
